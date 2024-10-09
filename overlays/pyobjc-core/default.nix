# umerged pr https://github.com/NixOS/nixpkgs/pull/336801
{
  lib,
  pkgs,
  stdenv,
  buildPythonPackage,
  setuptools,
  fetchPypi,
  xcodebuild,
  cctools,
  darwin,
}:
let
  appleSDK = darwin.apple_sdk_11_0;

  apple_libffi = stdenv.mkDerivation {
    pname = "apple-libffi";
    inherit (appleSDK.MacOSX-SDK) version;
    dontUnpack = true;
    installPhase = ''
      mkdir -p $out/include $out/lib
      cp -r ${appleSDK.MacOSX-SDK}/usr/include/ffi $out/include/
      cp -r ${appleSDK.MacOSX-SDK}/usr/lib/libffi.* $out/lib/
    '';
  };
in

buildPythonPackage rec {
  pname = "pyobjc-core";
  version = "10.3.1";
  src = fetchPypi {
    pname = "pyobjc_core";
    inherit version;
    hash = "sha256-sgSoDMwHD5qz+K9COjolpv14fiKFCNAMTDD4rFOLpyA=";
  };
  pyproject = true;
  build-system = [ setuptools ];

  nativeBuildInputs = [
    xcodebuild
    cctools
  ];

  buildInputs = [
    appleSDK.objc4
    appleSDK.frameworks.Foundation
    appleSDK.frameworks.GameplayKit
    appleSDK.frameworks.MetalPerformanceShaders
    apple_libffi
  ];

  checkPhase = ''
    # TODO: This library does not follow standard testing with pytest
    # and implemented its own test runner bootstrapping unittest
    python3 setup.py test
  '';

  hardeningDisable = [ "strictoverflow" ]; # -fno-strict-overflow is not supported in clang on darwin
  env.NIX_CFLAGS_COMPILE = toString [ "-Wno-error=deprecated-declarations" ];
  postPatch = ''
    # TODO: Make patch for setup.py
    # ignore the manual include flag for ffi, appears that it needs a very specific ffi from sdk (needs confirmation)
    substituteInPlace setup.py --replace-fail '"-I/usr/include/ffi"' '#"-I/usr/include/ffi"'
    # make os.path.exists that can spoil objc4 return True
    substituteInPlace setup.py --replace-fail 'os.path.join(self.sdk_root, "usr/include/objc/runtime.h")' '"/"'

    # Turn off clang’s Link Time Optimization, or else we can’t recognize (and link) Objective C .o’s:
    sed -r 's/"-flto=[^"]+",//g' -i setup.py
    # Fix some test code:
    grep -RF '"sw_vers"' | cut -d: -f1 | while IFS= read -r file ; do
      sed -r "s+"sw_vers"+"/usr/bin/sw_vers"+g" -i "$file"
    done

    # Disables broken tests and fixes some of them
    # TODO: make a patch for tests
    substituteInPlace \
      PyObjCTest/test_nsdecimal.py \
      --replace-fail "Cannot compare NSDecimal and decimal.Decimal" "Cannot compare NSDecimal and (\\\\w+.)?Decimal"
    substituteInPlace \
      PyObjCTest/test_bundleFunctions.py \
      --replace-fail "os.path.expanduser(\"~\")" "\"/var/empty\""
    substituteInPlace \
      PyObjCTest/test_methodaccess.py \
      --replace-fail "testClassThroughInstance2" \
      "disable_testClassThroughInstance2"


    # Fixes impurities in the package and fixes darwin min version
    # TODO: Propose a patch that fixes it in a better way
    # Force it to target our ‘darwinMinVersion’, it’s not recognized correctly:
    grep -RF -- '-DPyObjC_BUILD_RELEASE=%02d%02d' | cut -d: -f1 | while IFS= read -r file ; do
      sed -r '/-DPyObjC_BUILD_RELEASE=%02d%02d/{s/%02d%02d/${
        lib.concatMapStrings (lib.fixedWidthString 2 "0") (
          lib.splitString "." stdenv.targetPlatform.darwinMinVersion
        )
      }/;n;d;}' -i "$file"
    done
    # impurities:
    ( grep -RF '/usr/bin/xcrun' || true ; ) | cut -d: -f1 | while IFS= read -r file ; do
      sed -r "s+/usr/bin/xcrun+$(which xcrun)+g" -i "$file"
    done
    ( grep -RF '/usr/bin/python' || true ; ) | cut -d: -f1 | while IFS= read -r file ; do
      sed -r "s+/usr/bin/python+$(which python)+g" -i "$file"
    done

    # Adjust expected paths for libcrypto
    substituteInPlace PyObjCTest/test_dyld.py \
        --replace '/usr/lib/libcrypto.dylib' '${pkgs.openssl.out}/lib/libcrypto.dylib' \
        --replace '/Library/Frameworks/Python.framework/Versions/3.12/lib/libcrypto.dylib' '${pkgs.openssl.out}/lib/libcrypto.dylib'

    # Disable the failing test_dyld_framework test
    substituteInPlace PyObjCTest/test_dyld.py \
        --replace 'def test_dyld_framework' 'def disabled_test_dyld_framework'
  '';
  passthru = {
    inherit apple_libffi;
  };
  meta = {
    description = "The Python <-> Objective-C Bridge with bindings for macOS frameworks";
    homepage = "https://pypi.org/project/pyobjc-core/";
    platforms = lib.platforms.darwin;
    maintainers = [ lib.maintainers.ferrine ];
  };
}
