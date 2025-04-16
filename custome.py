class DBsingletonError(Exception):
    '''
    A custome error raised due to dbconnection 
    '''
    def __init__(self, *args):
        super().__init__(*args)
