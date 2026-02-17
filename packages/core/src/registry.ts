/**
 * Generic registry for pluggable implementations.
 */

export class Registry<T> {
  private readonly _map = new Map<string, () => T>();
  readonly subsystem: string;

  constructor(subsystem: string) {
    this.subsystem = subsystem;
  }

  register(name: string, factory: () => T): void {
    this._map.set(name, factory);
  }

  get(name: string): T {
    const factory = this._map.get(name);
    if (!factory) {
      const avail = [...this._map.keys()].join(", ");
      throw new Error(
        `[${this.subsystem}] Unknown implementation "${name}". Available: ${avail}`
      );
    }
    return factory();
  }

  has(name: string): boolean {
    return this._map.has(name);
  }

  list(): string[] {
    return [...this._map.keys()];
  }
}
