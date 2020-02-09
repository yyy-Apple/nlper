from CNN import CNNClassifier, train_embeddings, MyDataset, train_df, \
    val_df, test_df
import torch
from torch.utils.data import Dataset, DataLoader


def predict(val_or_test, fileout):
    myDataset = MyDataset.create_Dataset(train_df, val_df, test_df, 2)
    bestClassifier = CNNClassifier(train_embeddings, train_embeddings.shape[0], train_embeddings.shape[1],
                                   300, 100, len(myDataset.vectorizer.tagVocabulary), 0.5)
    # load the model
    bestClassifier = torch.nn.DataParallel(bestClassifier)
    bestClassifier.load_state_dict(torch.load('cnnClassifier.pth'))

    myDataset.set_target_df(val_or_test)

    predictions = []

    bestClassifier.eval()
    for i in range(len(myDataset.target_df)):
        if i % 10 == 0:
            print("on %d"%i,"data point")
        seq_and_tag = myDataset.__getitem__(i)
        seq = seq_and_tag['x']
        seq = torch.from_numpy(seq).unsqueeze(dim=0)
        pred_tag_idx = bestClassifier(seq, apply_softmax=True).argmax(dim=1).item()
        pred_tag = myDataset.vectorizer.tagVocabulary.index2word[pred_tag_idx]
        predictions.append(pred_tag)

    # write_to_file
    with open(fileout, "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")


if __name__ == "__main__":
    predict("val", "dev_results.txt")
    predict("test", "test_results.txt")
