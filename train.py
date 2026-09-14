from arguments import parse_training_config
from trainer import Trainer

def main():
    params = parse_training_config()
    trainer = Trainer(params)
    trainer.fit()

if __name__ == "__main__":
    main()
