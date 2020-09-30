def main():
    args = Parser().parse()

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader = get_data(args)

    # labels should be a whole number from [0, num_classes - 1]
    num_labels = 10 #int(max(max(train_data.targets), max(test_data.targets))) + 1
    output_size = num_labels
    setup_logging(args)

    if 'resnet' in args.model:
        constructor = getattr(resnet, args.model)
        model_stoch = constructor(True, device).to(device)
        model_det = constructor(False, device).to(device)

    elif 'vgg' in args.model:
        constructor = getattr(vgg, args.model)
        model_stoch = constructor(True, device, args.orthogonal).to(device)
        model_det = constructor(False, device, args.orthogonal).to(device)

    else:
        stoch_args = [True, True, device]
        det_args = [False, False, device]
        model_stoch = lenet5.LeNet5(*stoch_args).to(device)
        model_det = lenet5.LeNet5(*det_args).to(device)

    # load saved parameters
    saved_models = glob(f'experimental_models/{args.model}*')
    saved_det = saved_models[0] if 'det' in saved_models[0] else saved_models[1]
    saved_stoch = saved_models[1-saved_models.index(saved_det)]
    it = zip([model_stoch, model_det], [saved_stoch, saved_det])
    for model, param_path in it:
        saved_state = torch.load(param_path, map_location=device)
        if param_path[-4:] == '.tar':
            saved_state = saved_state['model_state_dict']
        model.load_state_dict(saved_state)


if __name__ == '__main__':
    main()
