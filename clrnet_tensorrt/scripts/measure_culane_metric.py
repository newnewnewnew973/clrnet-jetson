from runtime import PROJECT_ROOT, configure_import_paths, ensure_numpy_bool_alias


ensure_numpy_bool_alias()
configure_import_paths()

from clrnet_common.metric import main


if __name__ == "__main__":
    main("clrnet_tensorrt", PROJECT_ROOT)
