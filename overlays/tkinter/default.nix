{ lib
, stdenv
, buildPythonPackage
, python
, isPyPy
}:

buildPythonPackage {
  pname = "tkinter";
  version = python.version;
  format = "other";

  disabled = isPyPy;

  installPhase =
    ''
      # Move the tkinter module
      mkdir -p $out/${python.sitePackages}
      mv lib/${python.libPrefix}/lib-dynload/_tkinter* $out/${python.sitePackages}/
    ''
    + lib.optionalString (!stdenv.isDarwin) ''
      # Update the rpath to point to python without x11Support
      old_rpath=$(patchelf --print-rpath $out/${python.sitePackages}/_tkinter*)
      new_rpath=$(sed "s#${python}#${python}#g" <<< "$old_rpath" )
      patchelf --set-rpath $new_rpath $out/${python.sitePackages}/_tkinter*
    '';

  meta = python.meta // {
    description = "The standard Python interface to the Tcl/Tk GUI toolkit";
    longDescription = ''
      The tkinter package ("Tk interface") is the standard Python interface to
      the Tcl/Tk GUI toolkit. Both Tk and tkinter are available on most Unix
      platforms, including macOS, as well as on Windows systems.
    '';
  };
}
