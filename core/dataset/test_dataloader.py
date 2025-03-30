def print_loader(train_loader, val_loader, train_sampler, val_sampler, mg_sampler):
    print("=== DataLoader and Sampler Information ===")

    # 打印 train_loader 信息
    print("\n[Train Loader]")
    if train_loader:
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Dataset size: {len(train_loader.dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        print(f"Shuffle: {train_loader.shuffle if hasattr(train_loader, 'shuffle') else 'N/A'}")
    else:
        print("Train loader is None")

    print("\n[Validation Loader]")
    if val_loader:
        print(f"Batch size: {val_loader.batch_size}")
        print(f"Dataset size: {len(val_loader.dataset)}")
        print(f"Number of batches: {len(val_loader)}")
        print(f"Shuffle: {val_loader.shuffle if hasattr(val_loader, 'shuffle') else 'N/A'}")
    else:
        print("Validation loader is None")

    print("\n[Train Sampler]")
    if train_sampler:
        print(f"Sampler type: {type(train_sampler).__name__}")
        print(f"Sampler length: {len(train_sampler)}")
    else:
        print("Train sampler is None")

    print("\n[Validation Sampler]")
    if val_sampler:
        print(f"Sampler type: {type(val_sampler).__name__}")
        print(f"Sampler length: {len(val_sampler)}")
    else:
        print("Validation sampler is None")

    print("\n[MG Sampler]")
    if mg_sampler:
        print(f"Sampler type: {type(mg_sampler).__name__}")
        print(f"Sampler length: {len(mg_sampler)}")
    else:
        print("MG sampler is None")
