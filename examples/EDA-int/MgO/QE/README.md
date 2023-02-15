# i-PI and QuantumESPRESSO

```bash
/home/elia/Google-Drive/google-personal/q-e-develop/bin/pw.x < MgO.in > MgO.out --ipi localhost:UNIX &
```

```bash
/home/elia/Google-Drive/google-personal/i-pi-sabia/bin/i-pi input.xml
```

```bash
  <ffsocket mode='unix' name='pw' pbc='True'>
    <address>localhost</address>
  </ffsocket>
```

## vscode

```json
{
    "args": ["-inp","MgO.scf.in"]
}
```

## interface

https://www.quantum-espresso.org/Doc/pw_user_guide/node13.html

```bash
pw.x -in pw.input > pw.out --ipi localhost:UNIX
```
