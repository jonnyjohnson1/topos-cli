# default.nix
{ lib
, buildPythonPackage
, fetchFromGitHub
, fetchpatch
, pillow
, xlib
, six
, xvfb-run
, setuptools
, gobject-introspection
, pygobject3
, gtk3
, libayatana-appindicator
, stdenv
, pyobjc-core
}:

buildPythonPackage rec {
  pname = "pystray";
  version = "0.19.5";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "moses-palmer";
    repo = "pystray";
    rev = "v${version}";
    hash = "sha256-CZhbaXwKFrRBEomzfFPMQdMkTOl5lbgI64etfDRiRu4=";
  };

  patches = [
    (fetchpatch {
      url = "https://github.com/moses-palmer/pystray/commit/813007e3034d950d93a2f3e5b029611c3c9c98ad.patch";
      hash = "sha256-m2LfZcWXSfgxb73dac21VDdMDVz3evzcCz5QjdnfM1U=";
    })
  ];

  postPatch = ''
    substituteInPlace setup.py \
      --replace-fail "'sphinx >=1.3.1'" ""
  '';

  nativeBuildInputs = [
    gobject-introspection
    setuptools
  ];

  propagatedBuildInputs = [
    pillow
    six
    # pygobject3
    gtk3
    ] ++ lib.optionals stdenv.isDarwin [
      pyobjc-core
  ] ++ lib.optionals stdenv.isLinux [
    xlib
    libayatana-appindicator
  ];

  nativeCheckInputs = lib.optionals stdenv.isLinux [ xvfb-run ];

  checkPhase = lib.optionalString stdenv.isLinux ''
    rm tests/icon_tests.py # test needs user input

    xvfb-run -s '-screen 0 800x600x24' python setup.py test
  '';

  meta = with lib; {
    homepage = "https://github.com/moses-palmer/pystray";
    description = "This library allows you to create a system tray icon";
    license = with licenses; [
      gpl3Plus
      lgpl3Plus
    ];
    platforms = platforms.unix;
    maintainers = with maintainers; [ jojosch ];
  };
}
