from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='/data/dl_lecture_data'
dataType='val2014'
algName = 'fakecap'
annFile='%s/Evaluation-I/captions_eval-I.json'%(dataDir)
resFile='%s/Evaluation-I/captions_eval-I_%s_results.json'%(dataDir,algName)
evalImgsFile='./%s_evalImgs.json'%(algName)
evalFile='./%s_eval.json'%(algName)

# download Stanford models
#./get_stanford_models.sh

# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()

# print output evaluation scores
for metric, score in cocoEval.eval.items():
    print '%s: %.3f'%(metric, score)

# demo how to use evalImgs to retrieve low score result
#evals = [eva for eva in cocoEval.evalImgs if eva['CIDEr']<30]
#print 'ground truth captions'
#imgId = evals[0]['image_id']
#annIds = coco.getAnnIds(imgIds=imgId)
#anns = coco.loadAnns(annIds)
#coco.showAnns(anns)

#print '\n'
#print 'generated caption (CIDEr score %0.1f)'%(evals[0]['CIDEr'])
#annIds = cocoRes.getAnnIds(imgIds=imgId)
#anns = cocoRes.loadAnns(annIds)
#coco.showAnns(anns)

#img = coco.loadImgs(imgId)[0]
#I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
#plt.imshow(I)
#plt.axis('off')
#plt.show()

# plot score histogram
#ciderScores = [eva['CIDEr'] for eva in cocoEval.evalImgs]
#plt.hist(ciderScores)
#plt.title('Histogram of CIDEr Scores', fontsize=20)
#plt.xlabel('CIDEr score', fontsize=20)
#plt.ylabel('result counts', fontsize=20)
#plt.show()

# save evaluation results to ./results folder
json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
json.dump(cocoEval.eval,     open(evalFile, 'w'))