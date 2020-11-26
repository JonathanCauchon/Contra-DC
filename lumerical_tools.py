import sys, os, platform

# Lumerical Python API path on system

cwd = os.getcwd()

if platform.system() == 'Windows':
    try:
        lumapi_path = r'C:\\Program Files\\Lumerical\\v202\\api\\python'
        os.chdir(lumapi_path)
        sys.path.append(lumapi_path)
        import lumapi
    except FileNotFoundError:
        lumapi_path = r'C:/Program Files/Lumerical/FDTD/api/python'
        os.chdir(lumapi_path)
        sys.path.append(lumapi_path)
        import lumapi
        
else:
    lumapi_path = '/Applications/Lumerical/v202/api/python/'
    os.chdir(lumapi_path)
    sys.path.append(lumapi_path)
    import lumapi

dir_path = os.path.dirname(os.path.realpath(__file__))
if os.path.exists(os.path.join(lumapi_path,'lumapi.py')):
    print('Found lumapi path at' + ': ' +lumapi_path)
    sys.path.append(lumapi_path)
else:
    print('lumapi path does not exist, edit lumapi_path variable')
    
os.chdir(cwd)


def generate_dat(pol = 'TE', terminate = True):
    mode = lumapi.open('mode')
    
    # feed polarization into model
    if pol == 'TE':
        lumapi.evalScript(mode,"mode_label = 'TE'; mode_ID = '1';")
    elif pol == 'TM':
        lumapi.evalScript(mode,"mode_label = 'TM'; mode_ID = '2';")
        
    # run write sparams script
    lumapi.evalScript(mode,"cd('%s');"%dir_path)
    lumapi.evalScript(mode,'write_sparams;')
    
    if terminate == True:
        lumapi.close(mode)
    
    #run_INTC()
    return
