import os, ctypes
pkg = r'C:\workspace\voicecred\.conda\Lib\site-packages\torchcodec'
print('Listing torchcodec dir:', pkg)
for f in sorted(os.listdir(pkg)):
    if 'libtorchcodec_core' in f.lower() and f.lower().endswith('.dll'):
        path = os.path.join(pkg, f)
        print('\nTrying', path)
        try:
            ctypes.WinDLL(path)
            print('  -> Load OK')
        except Exception as e:
            print('  -> Load failed:', type(e).__name__, e)
print('\nDone')
