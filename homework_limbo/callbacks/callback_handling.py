
def import_callbacks(experiment_config):
    available_callbacks = os.listdir("callbacks/")
    available_callbacks = [i.replace(".py", "") for i in available_callbacks]

    imported_callbacks = []
    for callback_name in master_config.Core_Config.Save_Config.Callback.names:

        if callback_name in available_callbacks:
            callback_class_module = importlib.import_module("callbacks." + callback_name)
            callback_class = getattr(callback_class_module, callback_name)
        else:
            callback_class_module = importlib.import_module()

        callback = callback_class(master_config)
        imported_callbacks.append(callback)

    return imported_callbacks