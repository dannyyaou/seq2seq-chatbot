"""
Script for chatting with a trained chatbot model
"""

import datetime
from os import path
from bottle import route, run, template, request
import general_utils
import chat_command_handler
from chat_settings import ChatSettings
from chatbot_model import ChatbotModel
from vocabulary import Vocabulary

#Read the hyperparameters and configure paths
_, model_dir, hparams, checkpoint, _, _ = general_utils.initialize_session("chat")

#Load the vocabulary
print()
print("Loading vocabulary...")
if hparams.model_hparams.share_embedding:
    shared_vocab_filepath = path.join(model_dir, Vocabulary.SHARED_VOCAB_FILENAME)
    input_vocabulary = Vocabulary.load(shared_vocab_filepath)
    output_vocabulary = input_vocabulary
else:
    input_vocab_filepath = path.join(model_dir, Vocabulary.INPUT_VOCAB_FILENAME)
    input_vocabulary = Vocabulary.load(input_vocab_filepath)
    output_vocab_filepath = path.join(model_dir, Vocabulary.OUTPUT_VOCAB_FILENAME)
    output_vocabulary = Vocabulary.load(output_vocab_filepath)

# Setting up the chat
chatlog_filepath = path.join(model_dir, "chat_logs", "chatlog_{0}.txt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
chat_settings = ChatSettings(hparams.model_hparams, hparams.inference_hparams)
terminate_chat = False
reload_model = False
#Create the model
print()
print("Initializing model..." if not reload_model else "Re-initializing model...")
print()
with ChatbotModel(mode = "infer",
                  model_hparams = chat_settings.model_hparams,
                  input_vocabulary = input_vocabulary,
                  output_vocabulary = output_vocabulary,
                  model_dir = model_dir) as model:

    #Load the weights
    print()
    print("Loading model weights...")
    print()
    model.load(checkpoint)

    #Show the commands
    if not reload_model:
        chat_command_handler.print_commands()

    running = True;

    while running:
        running = False;
        #Get the input and check if it is a question or a command, and execute if it is a command
        @route('/chat')
        def index():
            question = request.query.question
            question_with_history, answer = model.chat(question, chat_settings)
            return template('answer: {{answer}}', answer=answer)

        run(host='localhost', port=300)
