import pefile, os
pkg = r'C:\workspace\voicecred\.conda\Lib\site-packages\torchcodec'
file = os.path.join(pkg, 'libtorchcodec_core7.dll')
if not os.path.exists(file):
    print('core7 not found at', file)
else:
    print('Inspecting', file)
    p = pefile.PE(file)
    dlls = set()
    for entry in getattr(p, 'DIRECTORY_ENTRY_IMPORT', []):
        dlls.add(entry.dll.decode('utf-8'))
    print('Imported DLLs (sample):')
    for d in sorted(dlls):
        print(' -', d)
print('done')
