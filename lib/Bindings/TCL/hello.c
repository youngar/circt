 #include <tcl.h>

 static int 
 Circttcl_Cmd(ClientData cdata, Tcl_Interp *interp, int objc, Tcl_Obj *const objv[])
 {
     Tcl_SetObjResult(interp, Tcl_NewStringObj("Hello, World!", -1));
     return TCL_OK;
 }

 int DLLEXPORT
 Circttcl_Init(Tcl_Interp *interp)
 {
     if (Tcl_InitStubs(interp, TCL_VERSION, 0) == NULL) {
         return TCL_ERROR;
     }
     /* changed this to check for an error - GPS */
     if (Tcl_PkgProvide(interp, "Hello", "1.0") == TCL_ERROR) {
         return TCL_ERROR;
     }
     Tcl_CreateObjCommand(interp, "hello", Circttcl_Cmd, NULL, NULL);
     return TCL_OK;
 }

