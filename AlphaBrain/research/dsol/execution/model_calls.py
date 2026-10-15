"""Serial, fail-closed call accounting; reserve durably before model dispatch."""
from ..data.artifacts import read, write


class ModelCallLedger:
    def __init__(self,path):
        self.path=path
        self.data=read(path)
        if self.data['reserved'] or self.data['calls']:
            raise ValueError('Automatic resume/retry is forbidden')

    def reserve(self,identity):
        d=self.data
        if d['inherited_reserved']+d['reserved']>=d['limit']:
            raise ValueError('Cumulative model call cap reached')
        index=d['reserved'];d['reserved']+=1
        d['calls'].append(dict(index=index,identity=identity,status='RESERVED'))
        write(self.path,d)
        return index

    def complete(self,index,artifact_sha256):
        row=self.data['calls'][index]
        if row['status']!='RESERVED':raise ValueError('Call already completed')
        row.update(status='COMPLETE',artifact_sha256=artifact_sha256)
        self.data['completed']+=1
        write(self.path,self.data)
