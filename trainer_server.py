import grpc
from concurrent import futures
import time
# import the generated classes :
import model_pb2
import model_pb2_grpc
# import functions
from trainer import optimize, train, predict

port = 8061


# create a class to define the server functions, derived from
class MoleculeTrainer(model_pb2_grpc.MoleculeTrainerServicer):
    def optimize(self, request, context):
        # define the buffer of the response :
        response = model_pb2.OptimizationOutput()
        response.hidden_channels, response.num_layers, response.num_heads, response.num_bases, \
        response.lr = optimize(
            request.data_filepath,
            request.smiles_col,
            request.target_col,
            request.batch_size,
            request.seed,
            request.gpu,
            request.n_calls,
            request.n_random_starts
        )
        return response

    def train(self, request, context):
        # define the buffer of the response :
        response = model_pb2.TrainingEnd()
        response.model_dir = train(
            request.data_filepath,
            request.smiles_col,
            request.target_col,
            request.batch_size,
            request.seed,
            request.gpu,
            request.hidden_channels,
            request.num_layers,
            request.num_heads,
            request.num_bases,
            request.lr,
            request.name,
        )
        return response

    def predict(self, request, context):
        # define the buffer of the response :
        response = model_pb2.Prediction()
        response.prediction = predict(
            request.model_directory,
            request.problem,
            request.target_col,
            request.smiles,
        )
        return response

# create a grpc server :
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
model_pb2_grpc.add_MoleculeTrainerServicer_to_server(MoleculeTrainer(), server)
print("Starting server. Listening on port : " + str(port))
server.add_insecure_port("[::]:{}".format(port))
server.start()
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
