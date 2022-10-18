import json
import boto3


bucket = 'aws-static-dataset-news'
def load_json_from_s3(file_name):
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, file_name)
    file_content = obj.get()['Body'].read().decode('utf-8')
    data = json.loads(file_content)

    return data


def write_json_to_s3(file_name, dict_object):
    s3 = boto3.client('s3')
    s3.put_object(
         Body=json.dumps(dict_object, ensure_ascii=False, indent='2'),
         Bucket=bucket,
         Key=file_name
    )