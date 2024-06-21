import enformer
import tensorflow as tf
from dataset import get_dataset
from train_utils import evaluate_extended_model

def main(check):
    model = enformer.Enformer(channels=1536,
                            num_heads=8,
                            num_transformer_layers=11,
                            pooling_type='max')

    extended_model = enformer.ExtendedEnformerMouse(model)

    checkpoint = tf.train.Checkpoint(module=extended_model)
    checkpoint.restore(check)

    for head in ['human', 'mouse']:
        test_dataset = get_dataset(head, 'test').batch(1).prefetch(2)

        metrics_test = evaluate_extended_model(extended_model,
                                    dataset=test_dataset,
                                    corr='PearsonR',
                                    head=head,
                                    max_steps=100)
        print(f'{head}_PearsonR')
        print({k: v.numpy().mean() for k, v in metrics_test.items()})

        metrics_test = evaluate_extended_model(extended_model,
                                    dataset=test_dataset,
                                    corr='R2',
                                    head=head,
                                    max_steps=100)
        print(f'{head}_R2')
        print({k: v.numpy().mean() for k, v in metrics_test.items()})

if __name__ == "__main__":
    print("Adam 50")
    main('checkpoints/extended_checkpoints/Adam-50_human_0.752153754234314-41')
    print("\nAdam 100")
    main('checkpoints/extended_checkpoints/Adam-100-2_epoch_0.7784038186073303-25')
    print("\nAdam 200")
    main('checkpoints/extended_checkpoints/Adam-200_human_0.7411980628967285-14')
    print("\nAdam 400")
    main('checkpoints/extended_checkpoints/Adam-400_human_0.74897700548172-8')