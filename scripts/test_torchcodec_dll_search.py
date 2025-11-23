import os, sys
print('Python version', sys.version.splitlines()[0])
# Candidate DLL directories
cand = []
# torch lib dir
torch_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
if os.path.isdir(torch_lib):
    cand.append(torch_lib)
# conda Library bin
conda_libbin = os.path.join(sys.prefix, 'Library', 'bin')
if os.path.isdir(conda_libbin):
    cand.append(conda_libbin)
# PATH entries mentioning ffmpeg
for p in os.environ.get('PATH','').split(os.pathsep):
    if 'ffmpeg' in p.lower():
        cand.append(p)

print('Trying to add DLL search dirs:')
for d in cand:
    try:
        os.add_dll_directory(d)
        print(' add_dll_directory OK:', d)
    except Exception as e:
        print(' add_dll_directory FAILED:', d, e)

# Try import
try:
    import torchcodec
    print('torchcodec imported OK', getattr(torchcodec,'__version__','?'))
except Exception as e:
    print('Failed to import torchcodec:', e)
    import traceback
    traceback.print_exc()

print('Done')
