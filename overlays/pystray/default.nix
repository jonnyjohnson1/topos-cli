{
  lib,
  buildPythonPackage,
  pythonOlder,
  fetchPypi,
  flit-core,
  pytestCheckHook,
}:

buildPythonPackage rec {
  pname = "pystray";
  version = "0.19.5";

  disabled = pythonOlder "3.7";

  format = "wheel";

  src = fetchPypi {
    inherit pname;
    inherit version;
    inherit format;
    dist="py2.py3";
    python="py2.py3";
    sha256 = "a0c2229d02cf87207297c22d86ffc57c86c227517b038c0d3c59df79295ac617";
    };

  meta = with lib; {
    changelog = "https://github.com/rasterio/affine/blob/${version}/CHANGES.txt";
    description = "Matrices describing affine transformation of the plane";
    license = licenses.bsd3;
    homepage = "https://github.com/rasterio/affine";
    maintainers = with maintainers; [ mredaelli ];
  };
}
