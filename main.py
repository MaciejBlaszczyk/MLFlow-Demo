from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow
import mlflow.sklearn
import mlflow.spark
import mlflow.keras
import keras as K
from keras import backend as KB
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from kafka import KafkaConsumer
from json import loads
from random import shuffle
import numpy as np
from math import isinf
import time


schema = StructType([
	StructField('bedrooms', FloatType(), True),
	StructField('bathrooms', FloatType(), True),
	StructField('floors', FloatType(), True),
	StructField('condition', FloatType(), True),
	StructField('grade', FloatType(), True),
	StructField('price', FloatType(), True)
])


consumer = KafkaConsumer(
    'house_data',
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='my-group',
     value_deserializer=lambda x: loads(x.decode('utf-8')))

	 
features = ["bedrooms", "bathrooms", "floors", "condition", "grade"]


def r2(y_true, y_pred):
    SS_res =  KB.sum(KB.square( y_true-y_pred ))
    SS_tot = KB.sum(KB.square( y_true - KB.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + KB.epsilon()) )

	
def root_mean_squared_error(y_true, y_pred):
        return KB.sqrt(KB.mean(KB.square(y_pred - y_true)))	
	
	
def merge_old_data_with(new_data):
	print('Merging new data')
	shuffle(new_data)
	new_test_part = np.array(new_data[:10])
	new_train_part = np.array(new_data[10:])
	if train_data.size == 0:
		return new_train_part, new_test_part
	else: 
		return np.concatenate((train_data, new_train_part)), np.concatenate((test_data, new_test_part))

		
def split_and_scale_data_for_MLP(train_data, test_data, counter):			
	print('----------Preparing data for MLP----------')
	standard_scaler = SklearnStandardScaler()
	
	X_train = train_data[:, :-1]
	y_train = train_data[:, -1]
	
	X_test = test_data[:, :-1]
	y_test = test_data[:, -1]	
	
	X_all = standard_scaler.fit_transform(np.concatenate((X_test, X_train)))
	X_test = X_all[:(10*counter)]
	X_train = X_all[(10*counter):]
	
	return X_train, y_train, X_test, y_test 
		
		
def create_MLP_model(hidden_neurons = 256, optimizer='adam'): 
	print('----------Creating MLP model----------')
	model = K.models.Sequential()
	model.add(K.layers.Dense(128, kernel_initializer='normal', input_shape=(5,), activation='relu'))
	model.add(K.layers.Dense(256, kernel_initializer='normal', activation='relu'))
	model.add(K.layers.Dense(1, kernel_initializer='normal', activation='linear'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[r2, root_mean_squared_error, 'mean_absolute_error'])
	return model

	
def create_spark_dataframes(train_data, test_data):
	print('----------Preparing spark dataframes----------')
	train_df = spark.createDataFrame(train_data.tolist(), schema=schema)
	test_df = spark.createDataFrame(test_data.tolist(), schema=schema)
	return train_df.select('price', *features), test_df.select('price', *features)


def prepare_spark_pipeline_for_DT():
	print('----------Preparing spark pipeline for DT----------')
	label_indexer = StringIndexer(inputCol="price", outputCol="label", handleInvalid="keep")
	vector_assembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")
	standard_scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
	DT_model = DecisionTreeRegressor(maxDepth=8)
	
	stages = [label_indexer, vector_assembler, standard_scaler, DT_model]
	pipeline = Pipeline(stages=stages)
	
	estimator_param = ParamGridBuilder().addGrid(DT_model.maxDepth, [8, 16]).addGrid(DT_model.impurity, ["variance"]).build()
	eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mse")
	return CrossValidator(estimator=pipeline, estimatorParamMaps=estimator_param, evaluator=eval, numFolds=3), eval

	
def calculate_and_log_metrics_for_DT(predictions, evaluator):
	print('----------Calculating metrics for DT----------')
	rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
	print("DT_RMSE: %.3f" % rmse)
	mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
	print("DT_MSE: %.3f" % mse)
	mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
	print("DT_MAE: %.3f" % mae)
	r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
	print("DT_R2: %.3f" % r2)
	return rmse, mse, mae, r2


def calculate_and_log_metrics_for_MLP(predictions):
	print('----------Calculating metrics for MLP----------')
	rmse = predictions.history['val_root_mean_squared_error'][-1]
	print("MLP_RMSE: %.3f" % rmse)
	mse = predictions.history['val_loss'][-1]
	print("MLP_MSE: %.3f" % mse)
	mae = predictions.history['val_mean_absolute_error'][-1]
	print("MLP_MAE: %.3f" % mae)
	r2 = predictions.history['val_r2'][-1]
	print("MLP_R2: %.3f" % r2)
	return rmse, mse, mae, r2	
	

spark = SparkSession.builder.getOrCreate()

train_data = np.array(list())
test_data = np.array(list())
new_data = list()
print('Ready to work')
counter = 0

for message in consumer:	
	message = message.value
	print('{}'.format(message))
	new_data.append(list(message.values()))
	if len(new_data) == 100:		
		train_data, test_data = merge_old_data_with(new_data)
		new_data = list()
		
		counter += 1
		X_train, y_train, X_test, y_test = split_and_scale_data_for_MLP(train_data, test_data, counter) 		
		MLP_model = create_MLP_model()		
		print('----------Training MLP----------')
		MLP_predictions = MLP_model.fit(X_train, y_train, batch_size = 32, epochs=300, verbose=0, validation_data=(X_test, y_test))	
		MLP_rmse, MLP_mse, MLP_mae, MLP_r2 = calculate_and_log_metrics_for_MLP(MLP_predictions)
		
		with mlflow.start_run(run_name='MLP_model'):	
			mlflow.log_metric("rmse", MLP_rmse)
			mlflow.log_metric("mse", MLP_mse)
			mlflow.log_metric("r2", MLP_r2)
			mlflow.log_metric("mae", MLP_mae)
			mlflow.keras.log_model(MLP_model, "MLP_model")
			current_time = str(time.time())
			mlflow.set_tag("model_name", "keras")
			mlflow.set_tag("model_details_name", "keras_" + current_time)
			mlflow.keras.save_model(MLP_model, "keras_" + current_time)
				
		train_dataframe, test_dataframe = create_spark_dataframes(train_data, test_data) 
		DT_model, DT_evaluator = prepare_spark_pipeline_for_DT()				
		print('----------Training DT----------')
		DT_model_fitted = DT_model.fit(train_dataframe)
		DT_prediction = DT_model_fitted.transform(test_dataframe)
		DT_rmse, DT_mse, DT_mae, DT_r2 = calculate_and_log_metrics_for_DT(DT_prediction, DT_evaluator)
		
		with mlflow.start_run(run_name='DT_model'):	
			mlflow.log_metric("rmse", DT_rmse)
			mlflow.log_metric("mse", DT_mse)
			if not isinf(DT_r2):
				mlflow.log_metric("r2", DT_r2)
			mlflow.log_metric("mae", DT_mae)
			mlflow.spark.log_model(DT_model_fitted.bestModel, "DT_model")
			current_time = str(time.time())
			mlflow.set_tag("model_name", "spark")
			mlflow.set_tag("model_details_name", "spark_" + current_time)
			mlflow.spark.save_model(DT_model_fitted.bestModel, "spark-model_" + current_time)





