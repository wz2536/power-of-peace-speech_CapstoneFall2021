## File Layout
- **scripts** contains training scripts of all models for launching SageMaker training job. 
- **Visualization** contains notebooks for producing performance and classification results visualization:
	- **visualize-performance.ipynb** contains codes for producing graphs in reports. Numbers in this file are copied directly from individual experiment .ipynb notebooks. 
	- **view-sentence.ipynb** uses LIME to visualize the learning result, highlights important keywords for some sample articles, and output the sample articles (not the LIME results) to an .txt file for record. Split-by-country is the experiment which results is visualized in this notebook. Images presented in the Final Report Section IV.III follows the similar idea, but that result is generated based on the Random Split experiment. The output images are not compatible with Sgaemaker notebooks, hence are not saved. One can re-run this notebook to view the outputs. Due to the randomness of LIME algorithm, the output might not 100% match with the ones presented in reports. But there is a large chance that the selected keywords overlaps with what is presented in the report.

- **Kfold Evaluation** contains codes to train each experiment setting 10 times and calculate mean and confidence interval for each performance metrics. All models in these notebooks are trained within notebook environment rather than launching the Sagemaker Training job for an easier Kfold dataset generation, performance evaluation, and comparison pipeline. Hence, the models are not saved on Sgaemaker server.
	- **fine-tune-roberta-minority-kfold.ipynb.ipynb** contains initial study to RoBERTa training hyperparameter turning (mentioned in Report 2), Kfold performance on original and shuffled sentences under Random Split and Split-by-country Setting.(Experiment mentioned in Final Report Section IV.I)
	- **fine-tune-roberta-country-balanced-KFold.ipynb** fine-tune RoBERTa with Country Balanced Approach on original and shuffled sentences. (Experiment mentioned in Final Report Section IV.II)
	- **all-models-comparison.ipynb** fine-tune RoBERTa, evaluated trained Logistic, XGBoost, and SVM on Random Split, Split-by-country, and Country-Balanced Random Split experiment setting. (Experiment mentioned in Final Report Section IV.IV: Side Notes). 

The following notebooks launched Sagemaker Training jobs to train the model on large sample sets, and save the trained model for later reference.  The pipeline presented in these notebooks are more suitable for production purpose.
- **fine-tune-roberta.ipynb** contains codes and execution records (saved model location) for fine-tuning RoBERTa model on original data by launching Sagemaker training job with script file from scripts.
- **fine-tune-roberta-shuffled.ipynb** contains codes and execution records (saved model location) for fine-tuning RoBERTa model on shuffled data by launching Sagemaker training job with script file from scripts.
- **logistic.ipynb** contains codes and execution records (performance) for logistic (fine-tuning RoBERTa struture itself) classifier on original sentence embeddings from fine-tuned RoBERTa.
- **xgboost.ipynb** contains codes and execution records (saved model location and performance evaluation on 1 trial) for training a XGBoost classifier by launchign Sagemaker training job using the saved original sentence embedding from fine-tuned RoBERTa as inputs. 
 

