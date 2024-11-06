{
  description = "topos-cli";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    process-compose-flake.url = "github:Platonic-Systems/process-compose-flake";
    services-flake.url = "github:juspay/services-flake";
  };

  outputs = { self, nixpkgs, flake-parts, poetry2nix, process-compose-flake, services-flake }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
        imports = [ inputs.process-compose-flake.flakeModule ];
        systems = [ "x86_64-linux" "aarch64-darwin" ];
        perSystem = { self', pkgs, system, lib, ... }:
        let
            pkgs = import nixpkgs {
                inherit system;
                overlays = [
                inputs.poetry2nix.overlays.default
                (final: prev: {
                    toposPoetryEnv = final.callPackage toposPoetryEnv { };
                    pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                    (python-final: python-prev: {
                        pystray = python-final.callPackage ./nix/overlays/pystray/default.nix { };
                    })
                    ];
                })
                ];
            };

            # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
            #TODO: Figure out how to add setuptools to all the packages which need it, this is currently not working as expected.
            overrides = pkgs.poetry2nix.overrides.withDefaults (final: super:
            pkgs.lib.mapAttrs
                (attr: systems: super.${attr}.overridePythonAttrs
                (old: {
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ map (a: final.${a}) systems;
                }))
                {
                # https://github.com/nix-community/poetry2nix/blob/master/docs/edgecases.md#modulenotfounderror-no-module-named-packagename
                package = [ "setuptools" ];
                }
            );
            toposPoetryEnv = pkgs.poetry2nix.mkPoetryEnv {
            projectDir = self;
            preferWheels = true;
            inherit overrides;
            };

            envFile = pkgs.writeText "env_dev" (builtins.readFile ./.env_dev);
            parseEnvFile = envFile:
            let
                content = builtins.readFile envFile;
                lines = lib.filter (l: l != "" && !lib.hasPrefix "#" l) (lib.splitString "\n" content);
                parseLine = l:
                let
                    parts = lib.splitString "=" l;
                in
                    { name = lib.head parts; value = lib.concatStringsSep "=" (lib.tail parts); };
            in
                builtins.listToAttrs (map parseLine lines);
            envVars = parseEnvFile ./.env_dev;

            configFile = pkgs.copyPathToStore ./config.yaml;
            yq = pkgs.yq-go;
            
            kafkaPreStartup = ''
            echo "Kafka is ready. Creating topic..."
            ${pkgs.apacheKafka}/bin/kafka-topics.sh --create --topic chat_topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 --if-not-exists
            '';

            # Note: This only loads the settings from the repos config file
            #        if one is not already set in the user's .config directory.
            toposSetupHook = ''
            export $(cat ${envFile} | xargs)
            export TOPOS_CONFIG_PATH="$HOME/.topos/config.yaml"
            mkdir -p "$(dirname "$TOPOS_CONFIG_PATH")"
            if [ ! -f "$TOPOS_CONFIG_PATH" ]; then
                echo "Creating new config file at $TOPOS_CONFIG_PATH"
                echo "# Topos Configuration" > "$TOPOS_CONFIG_PATH"
                ${yq}/bin/yq eval ${configFile} | while IFS= read -r line; do
                echo "$line" >> "$TOPOS_CONFIG_PATH"
                done
                echo "Config file created at $TOPOS_CONFIG_PATH"
            else
                echo "Config file already exists at $TOPOS_CONFIG_PATH"
            fi
            ${kafkaPreStartup}
            '';

        in
        {
            process-compose."services-flake-topos" = { config, ... }: {
                imports = [
                inputs.services-flake.processComposeModules.default
                (import ./nix/services/topos-service.nix { inherit pkgs lib config; topos = self'.packages.topos; })
                ];
                services = let dataDirBase = "$HOME/.topos"; in {
                    # Backend service to perform inference on LLM models
                    ollama."ollama" = {
                        enable = true;

                        # The models are usually huge, downloading them in every project
                        # directory can lead to a lot of duplication. Change here to a
                        # directory where the Ollama models can be stored and shared across
                        # projects.

                        # dataDir = "${dataDirBase}/ollama";

                        # Define the models to download when our app starts
                        #
                        # You can also initialize this to empty list, and download the
                        # models manually in the UI.
                        #
                        # Search for the models here: https://ollama.com/library
                        models = [ "dolphin-llama3" ];
                    };

                    postgres."pg" = {
                      # data options https://search.nixos.org/options?query=services.postgresql
                      enable = true;
                      package = pkgs.postgresql_16.withPackages (p: [ p.pgvector ]);
                      port = 5432;
                      listen_addresses = "127.0.0.1";
                      # dataDir = "${dataDirBase}/pg";
                      initialDatabases = [
                        { name = "${envVars.POSTGRES_DB}"; }
                      ];
                      
                      initialScript = {
                        before = ''
                          CREATE EXTENSION IF NOT EXISTS vector;
                          CREATE USER ${envVars.POSTGRES_USER} WITH SUPERUSER PASSWORD '${envVars.POSTGRES_PASSWORD}';
                        '';

                        after = ''
                        -- THESE CREATE TABLE STATEMENTS HERE DO NOT WORK
                        -- THE WAY THE TABLES GET BUILT RN IS THROUGH THE PYTHON CODE _ensure_table_exists

                          CREATE TABLE IF NOT EXISTS conversation (
                            message_id VARCHAR PRIMARY KEY,
                            conv_id VARCHAR NOT NULL,
                            userid VARCHAR NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            name VARCHAR,
                            role VARCHAR NOT NULL,
                            message TEXT NOT NULL
                        );

                        -- Create the utterance_token_info table
                        CREATE TABLE IF NOT EXISTS utterance_token_info (
                            message_id VARCHAR PRIMARY KEY,
                            conv_id VARCHAR NOT NULL,
                            userid VARCHAR NOT NULL,
                            name VARCHAR,
                            role VARCHAR NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            ents JSONB
                        );

                        -- Create the utterance_text_info table
                        CREATE TABLE IF NOT EXISTS utterance_text_info (
                            message_id VARCHAR PRIMARY KEY,
                            conv_id VARCHAR NOT NULL,
                            userid VARCHAR NOT NULL,
                            name VARCHAR,
                            role VARCHAR NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            moderator JSONB,
                            mod_label VARCHAR,
                            tern_sent JSONB,
                            tern_label VARCHAR,
                            emo_27 JSONB,
                            emo_27_label VARCHAR
                        );
                          GRANT ALL PRIVILEGES ON DATABASE ${envVars.POSTGRES_DB} TO ${envVars.POSTGRES_USER};
                          GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${envVars.POSTGRES_USER};
                          GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ${envVars.POSTGRES_USER};
                          GRANT ALL PRIVILEGES ON SCHEMA public TO ${envVars.POSTGRES_USER};

                          ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ${envVars.POSTGRES_USER};
                          ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ${envVars.POSTGRES_USER};
                        '';
                      };
                    };

                    zookeeper."zookeeper".enable = true;

                    apache-kafka."kafka" = {
                      enable = true;
                      port = 9092;
                      # dataDir = "${dataDirBase}/kafka";
                      settings = {
                        "offsets.topic.replication.factor" = 1;
                        "zookeeper.connect" = [ "localhost:2181" ];
                      };
                      formatLogDirs = true;
                      formatLogDirsIgnoreFormatted = true;
                      jvmOptions = [
                        "-Xmx512M"
                        "-Xms512M"
                      ];
                    };

                    topos.enable = true;
                    topos.args = [ "run" ];
                };
                settings.processes = {
                    kafka.depends_on."zookeeper".condition = "process_healthy";
                    kafka.depends_on.pg.condition = "process_healthy";
                    topos.depends_on.pg.condition = "process_healthy";
                    topos.depends_on.kafka.condition = "process_healthy";
                };
            };

            packages =  rec {
                toposPoetry = pkgs.poetry2nix.mkPoetryApplication {
                projectDir = self;
                preferWheels = true;
                inherit overrides;
                };
                topos = pkgs.writeShellScriptBin "topos" ''
                  ${toposSetupHook}
                  ${toposPoetry}/bin/topos "$@"
                '';
                default = self'.packages."services-flake-topos";
            };

            devShells = {
            # Shell for app dependencies.
            #
            #     nix develop
            #
            # Use this shell for developing your app.
            default = pkgs.mkShell {
                inputsFrom = [ toposPoetryEnv ];
                packages = [ ];
                shellHook = ''
                export PATH="${toposPoetryEnv}/bin:$PATH"
                ${toposSetupHook}
                '';
            };

            # Shell for poetry.
            #
            #     nix develop .#poetry
            #
            # Use this shell for changes to pyproject.toml and poetry.lock.
            poetry = pkgs.mkShell {
                packages = [ pkgs.poetry ];
            };
            };
            legacyPackages = pkgs;
        };
    };
}
