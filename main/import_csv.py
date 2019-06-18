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
    #print("Gli id: " +str(ids)+"\n")
    #print("I testi sono: "+str(texts)+"\n")
    #print("Gli user_id sono: "+str(user_ids)+"\n")
    #print("le date reciproche"+str(dates)+"\n")
    #print("il tipo di disastro è se 0 e 3 alluvione altrimenti terremoto "+str(kind_disaster)+"\n")
    #print("damage riportati o non o se è relativo oppure no"+str(classes)+"\n" )

    return texts





