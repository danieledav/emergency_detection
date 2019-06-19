import pandas as pd

def importCsv(file_csv):

    ids = []
    texts = []
    user_ids = []
    dates = []
    classes = []
    kind_disaster = []
    df = pd.read_csv(file_csv, delimiter=',')
    data = pd.DataFrame(df, columns=['id', 'text', 'user_id', 'created_at', 'disaster', 'class'])

    texts = list(data['text'])
    ids = list(data['id'])
    user_ids = list(data['user_id'])
    dates = list(data['created_at'])
    kind_disaster = list(data['disaster'])
    classes = list(data['class'])

    result=list()
    result.append(ids)
    result.append(texts)
    result.append(user_ids)
    result.append(dates)
    result.append(kind_disaster)
    result.append(classes)


    return result





