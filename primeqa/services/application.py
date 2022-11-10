from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.server import GrpcServer
from primeqa.services.rest_server.server import RestServer

if __name__ == "__main__":
    config = Settings()

    if config.mode == "grpc":
        grpc_server = GrpcServer(config=config)
        grpc_server.run()

    if config.mode == "rest":
        rest_server = RestServer(config=config)
        rest_server.run()
