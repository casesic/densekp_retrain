"""Small dependency-free Chrome DevTools smoke for the local editor."""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class CDP:
    def __init__(self, endpoint):
        address=urlsplit(endpoint)
        self.sock=socket.create_connection((address.hostname,address.port),timeout=10)
        self.sock.settimeout(20)
        key=base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f'GET {address.path} HTTP/1.1\r\nHost: {address.hostname}:{address.port}\r\n'
                           f'Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n'
                           'Sec-WebSocket-Version: 13\r\n\r\n').encode())
        response=b''
        while b'\r\n\r\n' not in response:response+=self.sock.recv(4096)
        if not response.startswith(b'HTTP/1.1 101 '):raise RuntimeError(response[:500])
        self.pending=response.split(b'\r\n\r\n',1)[1]
        self.ident=0

    def _read(self,n):
        while len(self.pending)<n:self.pending+=self.sock.recv(max(4096,n-len(self.pending)))
        chunk,self.pending=self.pending[:n],self.pending[n:]
        return chunk

    def send(self,method,params=None):
        self.ident+=1;ident=self.ident
        data=json.dumps(dict(id=ident,method=method,params=params or {})).encode()
        mask=os.urandom(4)
        header=bytes([0x81,0x80|len(data)]) if len(data)<126 else bytes([0x81,0x80|126])+struct.pack('!H',len(data))
        self.sock.sendall(header+mask+bytes(v^mask[i%4] for i,v in enumerate(data)))
        while True:
            first=self._read(2);size=first[1]&127
            if size==126:size=struct.unpack('!H',self._read(2))[0]
            elif size==127:size=struct.unpack('!Q',self._read(8))[0]
            if first[1]&128:
                key=self._read(4);payload=bytes(v^key[i%4] for i,v in enumerate(self._read(size)))
            else:payload=self._read(size)
            if first[0]&15!=1:continue
            event=json.loads(payload)
            if event.get('id')==ident:
                if 'error' in event:raise RuntimeError(event['error'])
                return event['result']

    def evaluate(self,expression):
        result=self.send('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
        if 'exceptionDetails' in result:raise RuntimeError(result['exceptionDetails'])
        return result['result'].get('value')

    def drag(self,point,dx,dy):
        x,y=point
        for kind,xx,yy,buttons in [('mousePressed',x,y,1),('mouseMoved',x+dx,y+dy,1),('mouseReleased',x+dx,y+dy,0)]:
            self.send('Input.dispatchMouseEvent',dict(type=kind,x=xx,y=yy,button='left',buttons=buttons,clickCount=1))


def main():
    request=Request('http://127.0.0.1:9222/json/new?http://127.0.0.1:8765/',method='PUT')
    page=json.load(urlopen(request,timeout=10));cdp=CDP(page['webSocketDebuggerUrl'])
    for _ in range(40):
        if cdp.evaluate('document.querySelector("#people")?.children.length')==2:break
        time.sleep(.25)
    else:raise RuntimeError('editor_did_not_load')
    before=cdp.evaluate('recipe.people[0].control_points_xy_m[0]')
    point=cdp.evaluate('(()=>{let p=toScreen(recipe.people[0].control_points_xy_m[0]),r=document.querySelector("#map").getBoundingClientRect();return [r.left+p[0],r.top+p[1]]})()')
    cdp.drag(point,3,0)
    after=cdp.evaluate('recipe.people[0].control_points_xy_m[0]')
    assert after[0]>before[0],dict(before=before,after=after,point=point,
                                    target=cdp.evaluate(f'document.elementFromPoint({point[0]},{point[1]})?.id'))
    cdp.evaluate('document.querySelector("#rigMode").click()')
    before_rig=cdp.evaluate('recipe.rigs[0].origin_m')
    point=cdp.evaluate('(()=>{let p=toScreen(recipe.rigs[0].origin_m),r=document.querySelector("#map").getBoundingClientRect();return [r.left+p[0],r.top+p[1]]})()')
    cdp.drag(point,3,0)
    after_rig=cdp.evaluate('recipe.rigs[0].origin_m')
    assert after_rig[0]>before_rig[0],dict(before=before_rig,after=after_rig,point=point)
    cdp.evaluate('(()=>{let t=document.querySelector("#timeline");t.value=14;t.dispatchEvent(new Event("input"))})()')
    assert cdp.evaluate('frame')==14
    cdp.evaluate('(()=>{let t=document.querySelector("#recipeId");t.value="browser_smoke_a3";t.dispatchEvent(new Event("change"))})()')
    cdp.evaluate('document.querySelector("#save").click()')
    for _ in range(40):
        status=cdp.evaluate('document.querySelector("#status").textContent')
        if status.startswith('已保存'):break
        time.sleep(.25)
    else:raise RuntimeError('save_failed:'+status)
    saved=cdp.evaluate('document.querySelector("#loadId").value')
    cdp.evaluate('document.querySelector("#load").click()')
    for _ in range(40):
        status=cdp.evaluate('document.querySelector("#status").textContent')
        if status.startswith('已载入'):break
        time.sleep(.25)
    else:raise RuntimeError('load_failed:'+status)
    cdp.evaluate('document.querySelector("#check").click()')
    for _ in range(40):
        status=cdp.evaluate('document.querySelector("#checkResult").textContent')
        if status:break
        time.sleep(.25)
    assert '通过' in status,status
    print(json.dumps(dict(status='passed',route_before=before,route_after=after,rig_before=before_rig,
                          rig_after=after_rig,frame=14,saved_recipe=saved,check=status),ensure_ascii=False))


if __name__=='__main__':main()
