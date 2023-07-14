import glob
import pickle

def load_graph(dataset):
    validation_graphs = {}
    if dataset == "original":
        files = glob.glob('./database/_hoaxy*.pkl')
        for file in files:
            try:
                validation_graph = pickle.load(open(file,'rb'))
                validation_graphs[file] = validation_graph
            except Exception as e:
                print(e)
                pass
    else:
        X_train, X_test = pickle.load(open('./database/split_hoaxy.pkl', 'rb'))
        print("X_train", len(X_train))
        print("X_test", len(X_test))

        if dataset == "train":
            files = X_train
        else:
            files = X_test

        for file in files:
            try:
                if file not in validation_graphs:
                    validation_graph = pickle.load(open(file,'rb'))
                    validation_graphs[file] = validation_graph
                else:
                    print("ALREADY LOADED", file)
            except Exception as e:
                print("ERROR", e)
                pass
    
    print("Loaded ", len(validation_graphs), " real graphs")
    return validation_graphs
