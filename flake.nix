{
  description = "topos";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachSystem ["aarch64-darwin"] (system:
      let
        pkgs = import nixpkgs {
            inherit system;
            overlays = [
            poetry2nix.overlays.default
            (final: prev: {
                myapp = final.callPackage myapp { };
                pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                  (python-final: python-prev: {
                    pystray = python-final.callPackage ./overlays/pystray/default.nix { };
                  })
                ];
            })
            ];
        };

        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        myapp = { poetry2nix, lib }: poetry2nix.mkPoetryApplication {
          projectDir = self;
          preferWheels = true;
          overrides = poetry2nix.overrides.withDefaults (final: super:
            lib.mapAttrs
              (attr: systems: super.${attr}.overridePythonAttrs
                (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ map (a: final.${a}) systems;
                }))
              {
                # https://github.com/nix-community/poetry2nix/blob/master/docs/edgecases.md#modulenotfounderror-no-module-named-packagename
                package = [ "setuptools" ];
              }
          );
        };

        envFile = pkgs.writeText "env_dev" (builtins.readFile ./.env_dev);
        configFile = pkgs.copyPathToStore ./config.yaml;
        yq = pkgs.yq-go;

        # Note: This only loads the settings from the repos config file
        #        if one is not already set in the user's .config directory.
        toposSetupHook = ''
          export TOPOS_CONFIG_PATH="$HOME/.config/topos/config.yaml"
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
        '';

        postgresSetupHook = ''
          # SETUP POSTGRES SERVER
          echo "Loading environment variables from Nix store"
          export $(cat ${envFile} | xargs)

          # Define PGDATA and LOGFILE based on environment variables
          export PGDATA=$(pwd)/pgdata
          LOGFILE=$PGDATA/postgresql.log

          echo "Initializing PostgreSQL data directory at $PGDATA"

          echo "PGDATA: $PGDATA"
          if [ ! -d "$PGDATA" ]; then
            mkdir -p "$PGDATA"
            initdb -D "$PGDATA"
          fi

          echo "Stopping any existing PostgreSQL server..."
          pg_ctl -D "$PGDATA" stop -s -m fast || echo "No existing server to stop."

          echo "Starting PostgreSQL server..."
          pg_ctl -D "$PGDATA" -l $LOGFILE start -w

          # Wait for PostgreSQL to start
          for i in {1..10}; do
            if pg_isready -q; then
              break
            fi
            echo "Waiting for PostgreSQL to start..."
            sleep 1
          done

          if ! pg_isready -q; then
            echo "Failed to start PostgreSQL. Check the logs at $LOGFILE"
            exit 1
          fi

          # Create the database if it doesn't exist
          if ! psql -lqt | cut -d \| -f 1 | grep -qw "$POSTGRES_DB"; then
            createdb "$POSTGRES_DB"
          fi

          # Create the user if they don't exist
          if ! psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='$POSTGRES_USER'" | grep -q 1; then
            createuser -s "$POSTGRES_USER"
          fi

          # Set up the test database, role, and tables
          psql -v ON_ERROR_STOP=1 -d $POSTGRES_DB <<SQL
          CREATE TABLE IF NOT EXISTS conversation_cache (
              conv_id TEXT PRIMARY KEY,
              message_data JSONB NOT NULL
          );

          CREATE TABLE IF NOT EXISTS entities (
              id TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              properties JSONB
          );

          CREATE TABLE IF NOT EXISTS relations (
              source_id TEXT,
              relation_type TEXT,
              target_id TEXT,
              properties JSONB,
              PRIMARY KEY (source_id, relation_type, target_id)
          );

          GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO $POSTGRES_USER;
          GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $POSTGRES_USER;
          GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $POSTGRES_USER;
          GRANT ALL PRIVILEGES ON SCHEMA public TO $POSTGRES_USER;

          ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $POSTGRES_USER;
          ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $POSTGRES_USER;

          -- If you're using PostgreSQL 14 or later, you can also add these:
          GRANT pg_read_all_data TO $POSTGRES_USER;
          GRANT pg_write_all_data TO $POSTGRES_USER;
        SQL

          echo "PostgreSQL setup complete. Logs can be found at $LOGFILE"
          # FINISH POSTGRES SERVER
        '';

        kafkaSetupHook = ''
          echo "Starting Kafka in Kraft mode..."

          # Set up necessary environment variables
          export KAFKA_HEAP_OPTS="-Xmx512M -Xms512M"
          export KAFKA_KRAFT_MODE=true
          echo ${pkgs.apacheKafka}

          # Prepare a default config for Kraft mode
          if [ ! -f ./kafka.properties ]; then
            echo "Initializing Kafka Kraft mode..."

            # Server 1 Kraft
            cp ${pkgs.apacheKafka}/config/kraft/server.properties ./server-1.properties
            sudo sed -i '57!s/PLAINTEXT/MQ/g' server-1.properties
            sudo sed -i '30s/.*/controller.quorum.voters=1@localhost:9091/' server-1.properties
            sudo sed -i '78s|.*|log.dirs=/tmp/kraft-combined-logs/server-1|' server-1.properties
            sudo sed -i '27s|.*|node.id=1|' server-1.properties
            sudo sed -i '42s|.*|listeners=MQ://:9092,CONTROLLER://:9091|' server-1.properties
            sudo sed -i '92s|.*|offsets.topic.replication.factor=1|' server-1.properties
            sudo sed -i '57s|.*|listener.security.protocol.map=CONTROLLER:PLAINTEXT,MQ:PLAINTEXT,SSL:SSL,SASL_PLAINTEXT:SASL_PLAINTEXT,SASL_SSL:SASL_SSL|' server-1.properties

          fi

          # Step 1
          KAFKA_CLUSTER_ID="$(${pkgs.apacheKafka}/bin/kafka-storage.sh random-uuid)"

          # Step 2
          ${pkgs.apacheKafka}/bin/kafka-storage.sh format -t $KAFKA_CLUSTER_ID -c ./server-1.properties

          # Step 3
          ${pkgs.apacheKafka}/bin/kafka-server-start.sh ./server-1.properties &

          # Step 4
          echo "Kafka environment is ready to use and running in detached terminals."

          # Step 5
          ${pkgs.apacheKafka}/bin/kafka-topics.sh --create --topic chat_topic --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

          sleep 3
        '';

      in
      {
        packages = {
            default = pkgs.myapp;
            topos = pkgs.writeShellScriptBin "topos" ''
            export PATH="${pkgs.myapp}/bin:$PATH"
            ${pkgs.myapp}/bin/topos run
            '';
        };

        devShells = {
          # Shell for app dependencies.
          #
          #     nix develop
          #
          # Use this shell for developing your app.
          default = pkgs.mkShell {
            inputsFrom = [  pkgs.myapp ];
            packages = [ pkgs.postgresql ];
            shellHook = ''
              export PATH="${pkgs.myapp}/bin:$PATH"
              ${toposSetupHook}
              ${postgresSetupHook}
            '';
          };

          # Shell for topos
          #
          #     nix develop .#topos
          #
          # Use this shell running topos
          topos = pkgs.mkShell {
            inputsFrom = [  pkgs.myapp ];
            packages = [ pkgs.postgresql ];
            shellHook = ''
              export PATH="${pkgs.myapp}/bin:$PATH"
              ${toposSetupHook}
              ${postgresSetupHook}
              topos run
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
      }
    );
}
