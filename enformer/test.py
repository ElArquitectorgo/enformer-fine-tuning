import enformer
import tensorflow as tf
from dataset import get_dataset
from train_utils import evaluate_model_R2, evaluate_model_PearsonR

def main(check):
    model = enformer.Enformer(channels=1536,
                            num_heads=8,
                            num_transformer_layers=11,
                            pooling_type='max')

    extended_model = enformer.ExtendedEnformer(model)

    checkpoint = tf.train.Checkpoint(module=extended_model)
    #latest = tf.train.latest_checkpoint('checkpoints')
    checkpoint.restore(check)

    test_dataset = get_dataset('human', 'test').batch(1).prefetch(2)
    metrics_test = evaluate_model_R2(extended_model,
                                dataset=test_dataset,
                                max_steps=100)
    print('R2')
    print({k: v.numpy().mean() for k, v in metrics_test.items()})

    metrics_test = evaluate_model_PearsonR(extended_model,
                                dataset=test_dataset,
                                max_steps=100)
    print('PearsonR')
    print({k: v.numpy().mean() for k, v in metrics_test.items()})

if __name__ == "__main__":
    print("Adam 50")
    main('checkpoints/Adam-50-10000_epoch_0.7562907338142395-45')
    print("Adam 100")
    main('checkpoints/Adam-100-10000_epoch_0.7498393654823303-26')
    print("Adam 200")
    main('checkpoints/Adam-200-10000_epoch_0.752016007900238-18')
    print("Adam 400")
    main('checkpoints/Adam-400-10000_epoch_0.7472643256187439-9')