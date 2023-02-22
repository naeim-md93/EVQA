


def remove_repeated_data(data):

    que_ans_img = {}

    for d in data:
        q = d['question']
        a = d['answer']
        i = d['image_id']
        d['repeated_data'] = False

        qai = f'{q}|||{a}|||{i}'

        if qai not in que_ans_img:
            que_ans_img[qai] = ''

        else:
            d['repeated_data'] = True

    data = [d for d in data if d['repeated_data'] is False]

    return data