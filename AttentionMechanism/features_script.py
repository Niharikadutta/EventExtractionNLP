import random

if __name__ == "__main__":
    filepath = "D:\Academic\ASU\Sem 4\\NLP\project\dataset\wiki_sentence_annotated.with_trigger.tsv"
    file = open(filepath, 'r', encoding='utf8')
    dict = {}
    numlines = 0
    for line in file:
        numlines += 1
        dict[numlines] = line
    file.close()
    # outfilepath = "D:\Academic\ASU\Sem 4\\NLP\project\dataset\smaller_features.txt"
    fileAbh = open("D:\Academic\ASU\Sem 4\\NLP\project\dataset\\featuresAbh.txt", "w", encoding='utf8')
    fileNat = open("D:\Academic\ASU\Sem 4\\NLP\project\dataset\\featuresNat.txt", "w", encoding='utf8')
    fileDip = open("D:\Academic\ASU\Sem 4\\NLP\project\dataset\\featuresDip.txt", "w", encoding='utf8')
    fileNih = open("D:\Academic\ASU\Sem 4\\NLP\project\dataset\\featuresNih.txt", "w", encoding='utf8')
    # outfile = open(outfilepath, 'w', encoding='utf8')
    print("number of lines: ", numlines)
    chosenlines = set()
    i = 0
    r = random.randint(1, 142610)
    while i < 14261:
        if r not in chosenlines:
            if i%4 == 0:
                fileAbh.write(dict[r])
            elif i%4 == 1:
                fileNat.write(dict[r])
            elif i%4 == 2:
                fileDip.write(dict[r])
            elif i%4 == 3:
                fileNih.write(dict[r])
            chosenlines.add(r)
            i += 1
            r = random.randint(1, 142610)
        else:
            r = random.randint(1, 142610)

    fileDip.close()
    fileNih.close()
    fileNat.close()
    fileAbh.close()
    # outfile.close()



