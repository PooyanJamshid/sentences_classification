import transformers
from function import model,device,weight_add_function
from logger_config import get_logger

#doeload model and tookenizer
model_albert=transformers.AlbertModel.from_pretrained('albert-base-v2')
tokenizer_albert=transformers.AlbertTokenizer.from_pretrained('albert-base-v2')

# for add weight file to model

#model_albert=weight_add_function(model_albert,'../models/weights_final_epoch_9.pth')

#build finally model
model_main_albert= model(model_albert,tokenizer_albert,768,10)

#make some layer trainable
for param in model_main_albert.parameters():
    param.requires_grad = False

for parameters in model_main_albert.model.encoder.embedding_hidden_mapping_in.parameters():
  parameters.requires_grad=True

for parameters in model_main_albert.model.encoder.albert_layer_groups[0].albert_layers[0].ffn.parameters():
  parameters.requires_grad=True

for parameters in model_main_albert.model.encoder.albert_layer_groups[0].albert_layers[0].ffn_output.parameters():
  parameters.requires_grad=True

#classification layer
for param in model_main_albert.classifier.parameters():
    param.requires_grad = True

logger = get_logger(__name__)
    

def log_model():
    logger.info('model is ready')
   