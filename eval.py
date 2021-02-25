#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, argparse, time
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import MNN
import onnxruntime

#from common.data_utils import preprocess_image

#from resnet import resnet_face18
#from torch.nn import DataParallel


def get_datapair_list(data_pair_path):
    '''
    load data pair file lines to form data pair list,
    which would be like:
    data_pair_list = [
        ['Abel_Pacheco/Abel_Pacheco_0001.jpg', 'Abel_Pacheco/Abel_Pacheco_0004.jpg', '1'],
        ['Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg', 'Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg', '1'],
        ...
        ['Abdel_Madi_Shabneh/Abdel_Madi_Shabneh_0001.jpg', 'Dean_Barker/Dean_Barker_0001.jpg', '0'],
        ...
    ]
    '''
    with open(data_pair_path, 'r') as fd:
        data_pairs = fd.readlines()

    data_pair_list = []
    for data_pair in data_pairs:
        data_pair = data_pair.split()
        assert len(data_pair) == 3, 'invalid data pair.'

        data_pair_list.append(data_pair)

    return data_pair_list


def merge_datapairs(data_pair_list):
    '''
    pick the image samples in data pair list and merge into single list (remove duplicated)
    for feature extract, merged list would be like:
    image_list = [
        'Abel_Pacheco/Abel_Pacheco_0001.jpg',
        'Abel_Pacheco/Abel_Pacheco_0004.jpg',
        'Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg'
        ...
    ]
    '''
    image_list_1 = [data_pair[0] for data_pair in data_pair_list]
    image_list_2 = [data_pair[1] for data_pair in data_pair_list]

    image_list = list(set(image_list_1 + image_list_2))

    return image_list


def cosine_metric(x1, x2):
    '''
    calculate cosine similarity for 2 vectors
    '''
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def load_image(image_path, model_input_shape):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, model_input_shape, interpolation=cv2.INTER_AREA)
    image = image[:, :, np.newaxis]
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


#def load_image(image_path, model_input_shape):
    ## prepare input image
    #image = Image.open(image_path).convert('RGB')
    #if image is None:
        #return None

    #image = preprocess_image(image, target_size=model_input_shape, return_tensor=False)
    #image = np.expand_dims(image, axis=0)
    #return image


def predict_torch(model, device, data):
    model.eval()
    with torch.no_grad():
        data = torch.from_numpy(data).to(device)
        feature = model(data).data.cpu().numpy()

    return feature


def predict_onnx(model, data):

    input_tensors = []
    for i, input_tensor in enumerate(model.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for image
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    feed = {input_tensors[0].name: data}
    feature = model.run(None, feed)

    return feature


def predict_mnn(interpreter, session, data):
    from functools import reduce
    from operator import mul

    # assume only 1 input tensor for image
    input_tensor = interpreter.getSessionInput(session)
    # get input shape
    input_shape = input_tensor.getShape()

    # use a temp tensor to copy data
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    input_elementsize = reduce(mul, input_shape)
    tmp_input = MNN.Tensor(input_shape, input_tensor.getDataType(),\
                    tuple(data.reshape(input_elementsize, -1)), input_tensor.getDimensionType())

    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    # we only handle single output model
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()

    assert output_tensor.getDataType() == MNN.Halide_Type_Float

    # copy output tensor to host, for further postprocess
    output_elementsize = reduce(mul, output_shape)
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

    output_tensor.copyToHostTensor(tmp_output)
    #tmp_output.printTensorData()

    output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)
    return output_data


def get_feature_dict(model, model_format, model_input_shape, device, dataset_path, image_list):
    '''
    run inference on every image to get feature vector and form up
    feature dict, would be like:
    feature_dict = {
        'Abel_Pacheco/Abel_Pacheco_0001.jpg' : array([-0.052, 0.307, ...]),
        'Abel_Pacheco/Abel_Pacheco_0004.jpg' : array([-0.0656, -0.211, ...]),,
        ...
    }
    '''
    if model_format == 'MNN':
        #MNN inference engine need create session
        session = model.createSession()

    feature_dict = {}

    pbar = tqdm(total=len(image_list), desc='Evaluating image features')
    for i, image_file in enumerate(image_list):
        pbar.update(1)
        image_data = load_image(os.path.join(dataset_path, image_file), model_input_shape)
        if image_data is None:
            print('read {} error'.format(image_file))
            continue

        # support of PyTorch pth model
        if model_format == 'PTH':
            feature = predict_torch(model, device, image_data)
        # support of ONNX model
        elif model_format == 'ONNX':
            feature = predict_onnx(model, image_data)
        # support of MNN model
        elif model_format == 'MNN':
            feature = predict_mnn(model, session, image_data)
        else:
            raise ValueError('invalid model format')

        feature_dict[image_file] = np.squeeze(feature)
    pbar.close()

    return feature_dict


def get_best_accuracy(sim_scores, labels):
    '''
    walk through the similarity list to get a best threshold
    which can make highest accuracy
    '''
    sim_scores = np.asarray(sim_scores)
    labels = np.asarray(labels)
    best_accuracy = 0
    best_threshold = 0

    for i in range(len(sim_scores)):
        # choose one score as threshold
        threshold = sim_scores[i]
        # check predict label under this threshold
        y_pred = (sim_scores >= threshold)
        accuracy = np.mean((y_pred == labels).astype(int))

        # record best accuracy and threshold
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return (best_accuracy, best_threshold)



def compute_accuracy(feature_dict, data_pair_list):
    '''
    calculate cosine similarity for 2 images in data pair,
    using feature vector get with model

    feature dict, would be like:
    feature_dict = {
        'Abel_Pacheco/Abel_Pacheco_0001.jpg' : array([-0.052, 0.307, ...]),
        'Abel_Pacheco/Abel_Pacheco_0004.jpg' : array([-0.0656, -0.211, ...]),,
        ...
    }
    '''
    similarity_list = []
    labels = []
    for data_pair in data_pair_list:
        # pick feature vector for data pair images
        feature_1 = feature_dict[data_pair[0]]
        feature_2 = feature_dict[data_pair[1]]
        # convert label char ('0'/'1') to int
        label = int(data_pair[2])
        # cosine similarity for the 2 features
        sim = cosine_metric(feature_1, feature_2)

        # record similarity & label as list
        similarity_list.append(sim)
        labels.append(label)

    accuracy, threshold = get_best_accuracy(similarity_list, labels)
    return accuracy, threshold


def accuracy_for_data_pairs(model, model_format, model_input_shape, device, dataset_path, data_pair_list):
    image_list = merge_datapairs(data_pair_list)
    #print(feature_list.shape)
    feature_dict = get_feature_dict(model, model_format, model_input_shape, device, dataset_path, image_list)
    accuracy, threshold = compute_accuracy(feature_dict, data_pair_list)
    return accuracy, threshold



def evaluate(model, model_format, model_input_shape, device, dataset_path, data_pair_path):
    data_pair_list = get_datapair_list(data_pair_path)

    accuracy, threshold = accuracy_for_data_pairs(model, model_format, model_input_shape, device, dataset_path, data_pair_list)
    print('validation accuracy:', accuracy, 'best threshold:', threshold)
    return accuracy, threshold



def load_eval_model(model_path, device):
    # support of PyTorch pth model
    if model_path.endswith('.pth'):
        model = torch.load(model_path, map_location=device)
        model_format = 'PTH'

    # support of ONNX model
    elif model_path.endswith('.onnx'):
        model = onnxruntime.InferenceSession(model_path)
        model_format = 'ONNX'

    # support of MNN model
    elif model_path.endswith('.mnn'):
        model = MNN.Interpreter(model_path)
        model_format = 'MNN'

    else:
        raise ValueError('invalid model file')

    return model, model_format



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate CNN feature extractor model (pth/onnx/mnn) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to evaluation image dataset')

    parser.add_argument(
        '--data_pair_path', type=str, required=True,
        help='path to data pair definition file')

    #parser.add_argument(
        #'--classes_path', type=str, required=False,
        #help='path to class definitions, default=%(default)s', default=os.path.join('configs' , 'voc_classes.txt'))

    parser.add_argument(
        '--model_input_shape', type=str, required=False,
        help='model image input size as <height>x<width>, default=%(default)s', default='128x128')

    args = parser.parse_args()
    height, width = args.model_input_shape.split('x')
    args.model_input_shape = (int(height), int(width))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)

    # get eval model
    model, model_format = load_eval_model(args.model_path, device)

    #model = resnet_face18(use_se=False)
    #model = DataParallel(model)
    #model.load_state_dict(torch.load('resnet18_110.pth', map_location=device))
    #model_format = 'PTH'

    start = time.time()
    evaluate(model, model_format, args.model_input_shape, device, args.dataset_path, args.data_pair_path)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))

if __name__ == '__main__':
    main()
