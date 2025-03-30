def freeze_model_weights(model, freeze_until_layer=15):
    for i in range(6):
        for name, param in model.features[i].named_parameters():
            param.requires_grad = False
    freeze_until_layer = max(0, min(freeze_until_layer, 15))
    for j in range(freeze_until_layer):
        for name, param in model.features[6][j].named_parameters():
            param.requires_grad = False