from OilOps._FUNCS_ import *

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def Find_Doc_Template(fpath = '.'):
    for f in os.listdir(fpath):
        if '.DOC' in f.upper():
            FILE = f
    return FILE

def Find_FileMatch(fpath = '.', TXT=None, CASE = False, regex = False):
    if regex:
        FILE = []
        for f in os.listdir(fpath):
            if CASE:
                x = re.match(rf'.*{TXT}.*',f)
            else:
                x = re.match(rf'.*{TXT}.*',f, re.I)
            try:
                FILE.append(x.string)   
            except:
                pass
        if len(FILE) == 0:
            return None
        else:
            return FILE[0]
    else:
        TXT = str(TXT)
        if not CASE:
            TXT = TXT.upper()
        for f in os.listdir(fpath):
            if not CASE:
                f = f.upper()
            if TXT in f.upper():
                FILE = f
        return FILE


def Create_Terms_Form(text,FORMFILE = None):
    terms = re.findall(re.compile(r'\${([\w]*)}',re.I),text)
    terms = list(set(terms))
    if FORMFILE== None:
        FORMFILE = 'FORM_TERMS' + '_'+datetime.datetime.now().strftime('%Y%m%d') + '.xlsx'
    if os.path.exists(FORMFILE):
        FF = FORMFILE.split('.')

        if datetime.datetime.now().strftime('%Y%m%d') in FORMFILE:
            FF[0] = FF[0].replace(datetime.datetime.now().strftime('%Y%m%d'),datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        else:
            FF[0] = FF[0]+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')   
        FORMFIlE = FF[0]+FF[1]
    print('FORMFILE: '+ FORMFILE)
    pd.DataFrame({'TERMS':terms,'VALUES':None}).to_excel(FORMFILE)

def Load_Terms_Form(FILE='FORM_TERMS.xlsx'):
    df_read = pd.read_excel(FILE)
    df_read['VALUES'] = df_read['VALUES'].astype(str)
    D = df_read.set_index('TERMS',drop=True)['VALUES'].to_dict()
    return D

def Clear_Highlights(doc_in):
    for p in doc_in.paragraphs:
        for r in p.runs:
            r.font.highlight_color=None
    return doc_in
    
def Replace_Doc_Text(DOCNAME,REP_DICT, unhighlight=True):
    doc = docx.Document(DOCNAME)
    if unhighlight:
        doc=Clear_Highlights(doc)
    docx_replace(doc, **REP_DICT)		
    DOCNAME2 =DOCNAME.split('.')
    DOCNAME2 = DOCNAME2[0]+'REPLACED.'+DOCNAME2[1]
    doc.save(DOCNAME2)

CO505(Template = False,Final = False):
    if Template:
        FILE = Find_Doc_Template()
        t = getText(FILE)
        Create_Terms_Form(t)
    if Final:
        F_FORM = Find_FileMatch('.','TESTIMONY.*\.DOC',False, True)
        files = filter(os.path.isfile, os.listdir())
        files = [f for f in files if 'FORM_TERMS' in  f.upper() and '.XLS' in f.upper()]
        files = [os.path.join(adir, f) for f in files] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x))
        F_TERMS_FILE = files[-1]
        #Pergamos 0315 Geology Testimony JAB_FORM TERMS.xlsx'
        FORM_DICT = Load_Terms_Form(F_TERMS_FILE)
        Replace_Doc_Text(F_FORM,FORM_DICT)
    return None
      
