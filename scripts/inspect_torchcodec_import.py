import traceback
try:
    import torchcodec
    print('torchcodec imported, version:', getattr(torchcodec, '__version__', 'unknown'))
    import os
    pkgdir = os.path.dirname(torchcodec.__file__)
    print('torchcodec package dir:', pkgdir)
    print('lib files present:', [f for f in os.listdir(pkgdir) if 'libtorchcodec' in f.lower()])
except Exception as e:
    print('Exception during import:')
    traceback.print_exc()
