import settings
from loader.model_loader import loadmodel
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean

fo = FeatureOperator()
model = loadmodel(hook_feature)

############ STEP 1: feature extraction ###############
features, maxfeature = fo.feature_extraction(model=model)
print('End of STEP 1')

for layer_id,layer in enumerate(settings.FEATURE_NAMES):
############ STEP 2: calculating threshold ############
    thresholds = fo.quantile_threshold(features[layer_id],savepath=f"quantile_{layer}.npy")
    print('End of STEP 2')

############ STEP 3: calculating IoU scores ###########
    tally_result = fo.tally(features[layer_id],thresholds,savepath=f"tally_{layer}.csv")
    print('End of STEP 3')

############ STEP 4: generating results ###############
    generate_html_summary(fo.data, layer,
                          tally_result=tally_result,
                          maxfeature=maxfeature[layer_id],
                          features=features[layer_id],
                          thresholds=thresholds)
    print('End of STEP 4')
if settings.CLEAN:
    clean()