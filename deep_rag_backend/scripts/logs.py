#!/usr/bin/env python3
"""
Dynamic Docker Compose logs script for backend service.
Captures logs from a specific service and writes to a file.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def get_logs(service: str, tail: int = 500, follow: bool = False, output_file: str = None):
    """
    Get logs from a Docker Compose service.
    
    Args:
        service: Service name (e.g., 'api', 'db', 'frontend')
        tail: Number of lines to show (default: 500)
        follow: Whether to follow logs in real-time (default: False)
        output_file: Output file path (default: {service}_logs.txt)
    """
    # Determine compose file path (from backend directory, go up to root)
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    project_root = backend_dir.parent
    compose_file = project_root / "docker-compose.yml"
    
    if not compose_file.exists():
        print(f"Error: docker-compose.yml not found at {compose_file}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file
    if output_file is None:
        output_file = f"{service}_logs.txt"
    
    output_path = Path(output_file)
    
    # Build docker compose command
    cmd = [
        "docker", "compose",
        "-f", str(compose_file),
        "logs",
        "--tail", str(tail)
    ]
    
    if follow:
        cmd.append("-f")
    
    cmd.append(service)
    
    print(f"Capturing logs from service '{service}' (last {tail} lines)...")
    if follow:
        print(f"Following logs in real-time. Output: {output_path}")
    else:
        print(f"Output: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            
            if follow:
                print("Press Ctrl+C to stop following logs...")
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\nStopping log capture...")
                    process.terminate()
                    process.wait()
            else:
                process.wait()
        
        if process.returncode == 0:
            print(f"✓ Logs saved to {output_path}")
        else:
            print(f"⚠ Warning: Process exited with code {process.returncode}", file=sys.stderr)
            sys.exit(process.returncode)
            
    except FileNotFoundError:
        print("Error: 'docker compose' command not found. Is Docker installed?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Capture Docker Compose service logs to a file (backend service)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture last 500 lines from api service
  python scripts/logs.py api
  
  # Capture last 1000 lines from db service to custom file
  python scripts/logs.py db --tail 1000 --output db_debug.log
  
  # Follow logs in real-time from frontend service
  python scripts/logs.py frontend --follow
        """
    )
    
    parser.add_argument(
        "service",
        help="Service name (e.g., 'api', 'db', 'frontend')"
    )
    
    parser.add_argument(
        "--tail",
        type=int,
        default=500,
        help="Number of lines to show (default: 500)"
    )
    
    parser.add_argument(
        "--follow", "-f",
        action="store_true",
        help="Follow logs in real-time (like tail -f)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: {service}_logs.txt)"
    )
    
    args = parser.parse_args()
    get_logs(
        service=args.service,
        tail=args.tail,
        follow=args.follow,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

