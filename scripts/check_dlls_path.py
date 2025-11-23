import os
paths=os.environ.get('PATH','').split(os.pathsep)
candidates=['MSVCP140.dll','VCRUNTIME140.dll','MSVCP140_1.dll']
found={}
for p in paths:
    for name in candidates:
        f=os.path.join(p,name)
        if os.path.exists(f):
            found.setdefault(name,[]).append(f)
print(found)
print('\nSystem32 check:')
for name in candidates:
    for d in [os.path.join(os.environ.get('WINDIR','C:\\Windows'),'System32'),os.path.join(os.environ.get('WINDIR','C:\\Windows'),'SysWOW64')]:
        p=os.path.join(d,name)
        if os.path.exists(p):
            print(name,'->',p)
