from src.datasets.data_pipeline import get_train_val_test_loaders

from src.utils.config import GPTConfig
config = GPTConfig()

train_loader, val_loader, test_loader = get_train_val_test_loaders(config)

train_x, train_y = next(iter(train_loader))
val_x, val_y = next(iter(val_loader))
test_x, test_y = next(iter(test_loader))

print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
print(test_x.shape, test_y.shape)