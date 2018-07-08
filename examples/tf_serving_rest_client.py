import json
import requests


def get_rest_url(model_name, host='127.0.1', port='8501', verb='predict', version=None):
    url = "http://{host}:{port}/v1/models/{model_name}".format(host=host, port=port, model_name=model_name)
    if version:
        url += 'versions/{version}'.format(version=version)
    url += ':{verb}'.format(verb=verb)
    return url


def get_model_prediction(model_input, model_name='amazon_review', signature_name='serving_default'):
    """ no error handling at all, just poc"""

    url = get_rest_url(model_name)
    data = {"instances": [model_input.tolist()]}

    rv = requests.post(url, data=json.dumps(data))
    if rv.status_code != requests.codes.ok:
        rv.raise_for_status()
    
    return rv.json()['predictions']

if __name__ == '__main__':
    from sample_model.utils import clean_data_encoded

    print("\nGenerate REST url ...")
    url = get_rest_url(model_name='amazon_review')
    print(url)
    
    while True:
        print("\nEnter an Amazon review [:q for Quit]")
        sentence = input()
        if sentence == ':q':
            break
        model_input = clean_data_encoded(sentence)
        model_prediction = get_model_prediction(model_input)
        print("The model predicted ...")
        print(model_prediction)