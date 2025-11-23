import os, sys, json
print('python:', sys.version.splitlines()[0])

# ffmpeg
try:
    import subprocess
    out = subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT, text=True)
    print('\nFFMPEG version output (first line):')
    print(out.splitlines()[0])
except Exception as e:
    print('\nffmpeg not available or error:', e)

# torchcodec
try:
    import torchcodec
    print('\ntorchcodec version:', getattr(torchcodec, '__version__', 'unknown'))
    pkgdir = os.path.dirname(torchcodec.__file__)
    print('torchcodec package dir:', pkgdir)
    dlls = [f for f in os.listdir(pkgdir) if f.lower().endswith('.dll')]
    print('dll files in torchcodec package:', dlls)
except Exception as e:
    print('\ntorchcodec import error:', e)

# torch
try:
    import torch
    print('\ntorch version:', torch.__version__)
except Exception as e:
    print('\ntorch import error:', e)

# check for libtorchcodec DLL in Library folders
site_lib = os.path.join(sys.prefix, 'Lib', 'site-packages')
print('\nsite-packages:', site_lib)
possible = []
if os.path.exists(site_lib):
    for root, dirs, files in os.walk(site_lib):
        for f in files:
            if 'libtorchcodec' in f.lower():
                possible.append(os.path.join(root, f))
print('Found libtorchcodec files (sample):', possible[:10])

# Quick advice string
print('\nNotes:')
print('- ffmpeg should be 7.1.1 for compatibility with libtorchcodec built for ffmpeg 7.x')
print("- If torchcodec shows version < 0.8.x and DLLs are missing, you can try upgrading torchcodec to 0.8.x via pip/conda and ensure ffmpeg 7.1.1 is on PATH")
print("- On Windows, prefer installing ffmpeg 7.1.1 via conda-forge or Chocolatey to get the necessary runtime DLLs")
