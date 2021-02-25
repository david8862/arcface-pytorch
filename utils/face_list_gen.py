#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Input path for the origin images package', type=str, required=True)
    parser.add_argument('--output_path', help='Output path for the picked image', type=str, required=True)
    args = parser.parse_args()

    class_names = os.listdir(args.input_path)

    output_file = open(args.output_path, 'w')

    for file_path, file_dir, files in os.walk(args.input_path):
        for checked_file in files:
            class_name = file_path.split(os.path.sep)[-1]
            class_id = class_names.index(class_name)

            print('checked_file', checked_file)
            print('class_name', class_name)
            print('class_id', class_id)

            output_file.write(os.path.join(class_name, checked_file) + ' ' + str(class_id))
            output_file.write('\n')

    output_file.close()


if __name__ == "__main__":
    main()

