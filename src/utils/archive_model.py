
"""archive model"""

import boto3
import os
import argparse
import json
import subprocess

NEXAR_DEEP_LEARNING_BUCKET = 'nexar-deep-learning'
MODEL_ARCHIVE_PATH = 'models/archives'


def check_if_file_exists(client, bucket, s3_path):
    try:
        client.head_object(Bucket=bucket, Key=s3_path)
        return True
    except:
        return False


def upload_dir_to_s3(local_directory, bucket, destination):
    file_in_folder = 0
    file_uploaded = 0
    file_skipped = 0
    print("Trying to upload files at [%s] to s3 {bucket[%s], destination[%s]}" % (local_directory, bucket, destination))
    client = boto3.client('s3')

    if not os.path.isdir(local_directory):
        raise Exception("[%s] not exists" % os.path.abspath(local_directory))

    # enumerate local files recursively
    for root, dirs, files in os.walk(local_directory):
        file_in_folder = len(files)

        for filename in files:

            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)

            # relative_path = os.path.relpath(os.path.join(root, filename))

            # print('Searching "%s" in "%s"' % (s3_path, bucket))
            if check_if_file_exists(client, bucket, s3_path):
                print("Path found on S3! Skipping [s3://%s/%s]..." % (bucket, s3_path))
                file_skipped += 1
            else:
                client.upload_file(local_path, bucket, s3_path)
                file_uploaded += 1
    print("Upload Status: file_in_folder[%s], file_uploaded[%s], file_skipped[%s]"
          % (file_in_folder, file_uploaded, file_skipped))


def generate_git_info_json():
    json_fname = 'git-info.json'
    if os.path.isfile(json_fname):
        os.remove(json_fname)
    current_script_dir = os.path.dirname(os.path.realpath(__file__))
    process = subprocess.Popen("git branch".split(), stdout=subprocess.PIPE, cwd=current_script_dir)
    output, error = process.communicate()
    branch = output.split("*")[1].split("\n")[0]
    process = subprocess.Popen("git rev-parse HEAD".split(), stdout=subprocess.PIPE, cwd=current_script_dir)
    output, error = process.communicate()
    commit = output.split("\n")[0]
    data = {'git_branch': branch, 'git_commit': commit}
    with open(json_fname, 'w') as outfile:
        json.dump(data, outfile)
    return json_fname


def argument_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dest", "--destination", action="store", default="",
                        help="destination path in s3://nexar-deep-learning/models/archives")
    parser.add_argument("-path", "--local_path", action="store", default="",
                        help="local path to upload to s3")
    args = parser.parse_args()
    return args


def upload_file(git_metadata_file, bucket, destination):
    client = boto3.client('s3')

    if not os.path.isfile(git_metadata_file):
        raise Exception("[%s] not exists" % os.path.abspath(git_metadata_file))

    s3_path = os.path.join(destination, os.path.basename(git_metadata_file))
    if check_if_file_exists(client, bucket, s3_path):
        raise Exception('file exists in S3 [s3://%s/%s]' % (bucket, s3_path))
    client.upload_file(git_metadata_file, bucket, s3_path)
    print("upload [%s] to [s3://%s/%s]" % (git_metadata_file, bucket, s3_path))


def main(args):
    if args.local_path == "" or args.destination == "":
        raise Exception("local_directory [empty=%r] or destination [empty=%r] can't be empty" %
                        (args.local_path == "", args.destination == ""))
    final_dir = MODEL_ARCHIVE_PATH + "/" + args.destination
    git_metadata_file = generate_git_info_json()
    upload_file(git_metadata_file, NEXAR_DEEP_LEARNING_BUCKET, final_dir)
    os.remove(git_metadata_file)
    upload_dir_to_s3(args.local_path, NEXAR_DEEP_LEARNING_BUCKET, final_dir)

if __name__ == "__main__":
    args = argument_setup()
    main(args)
