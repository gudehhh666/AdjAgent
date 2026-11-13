"""Host-owned immutable tool definitions; discovery is not authorization."""

from dataclasses import dataclass
import hashlib
import json
import re
from threading import RLock

from jsonschema import validators
from referencing import Registry
from referencing.exceptions import NoSuchResource


def canonical_json(value):
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("JSON object keys must be strings")
            for nested in item.values():
                check(nested)
        elif isinstance(item, list):
            for nested in item:
                check(nested)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise TypeError(f"Not a JSON value: {type(item).__name__}")
    check(value)
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # SQLite, fingerprints and MCP transports require actual UTF-8, not lone
    # surrogate code points that Python's JSON encoder otherwise tolerates.
    serialized.encode("utf-8")
    return serialized


def _no_retrieval(uri):
    raise NoSuchResource(ref=uri)


def validate_schema(schema):
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"}:
                    if not isinstance(item, str) or not item.startswith("#"):
                        raise ValueError("Only local JSON-schema references are allowed")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    try:
        canonical_json(schema)
        walk(schema)
        validators.validator_for(schema).check_schema(schema)
    except Exception as exc:
        raise ValueError(f"Invalid JSON schema: {exc}") from exc


def validator(schema):
    """No schema validation operation may retrieve remote resources."""
    return validators.validator_for(schema)(schema, registry=Registry(retrieve=_no_retrieval))


@dataclass(frozen=True, init=False, slots=True)
class ToolSpec:
    name: str
    description: str
    _input_json: str
    _output_json: str | None
    version: str
    network: bool
    requires_approval: bool

    def __init__(self, name, description, input_schema, output_schema=None,
                 version="1", network=False, requires_approval=False):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Tool name must be a model-compatible identifier")
        if not isinstance(description, str) or not isinstance(version, str):
            raise ValueError("Tool description and version must be strings")
        canonical_json([description, version])
        if not isinstance(input_schema, dict):
            raise ValueError("Tool input schema must be an object schema")
        if type(network) is not bool or type(requires_approval) is not bool:
            raise ValueError("Risk metadata must be host-configured booleans")
        validate_schema(input_schema)
        if output_schema is not None:
            validate_schema(output_schema)
        for key, value in {"name": name, "description": description,
                           "_input_json": canonical_json(input_schema),
                           "_output_json": None if output_schema is None else canonical_json(output_schema),
                           "version": version, "network": network, "requires_approval": requires_approval}.items():
            object.__setattr__(self, key, value)

    @property
    def input_schema(self):
        return json.loads(self._input_json)

    @property
    def output_schema(self):
        return None if self._output_json is None else json.loads(self._output_json)

    @property
    def definition_hash(self):
        definition = {"name": self.name, "description": self.description, "input_schema": self.input_schema,
                      "output_schema": self.output_schema, "version": self.version,
                      "network": self.network, "requires_approval": self.requires_approval}
        return hashlib.sha256(canonical_json(definition).encode()).hexdigest()

    def model_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.input_schema}}


@dataclass(frozen=True)
class ToolBinding:
    spec: ToolSpec
    handler: object


class ToolCatalog:
    def __init__(self):
        self._bindings = {}
        self._secrets = frozenset()
        self._lock = RLock()

    def register(self, spec, handler):
        self.register_many([(spec, handler)])

    def register_many(self, bindings, *, secrets=()):
        """Atomically publish definitions and host-only redaction metadata.

        Secrets remain memory-only, never ToolSpec/model/search/repr metadata.
        """
        bindings = list(bindings)
        if any(not isinstance(spec, ToolSpec) or not callable(handler) for spec, handler in bindings):
            raise TypeError("Registration requires ToolSpec definitions and callable adapters")
        if isinstance(secrets, (str, bytes)):
            raise TypeError("Secrets must be an iterable of host-configured strings")
        secrets = tuple(secrets)
        if any(not isinstance(value, str) for value in secrets):
            raise TypeError("Secret values must be strings")
        with self._lock:
            names = [spec.name for spec, _ in bindings]
            if len(names) != len(set(names)) or set(names) & self._bindings.keys():
                raise ValueError("Duplicate tool name in catalog registration")
            self._bindings.update({spec.name: ToolBinding(spec, handler) for spec, handler in bindings})
            self._secrets = self._secrets.union(value for value in secrets if value)

    def prepare_store(self, store):
        """Prime the actual execution Store before any provider-related journal write."""
        with self._lock:
            secrets = tuple(self._secrets)
        store.register_secrets(secrets)

    def get(self, name):
        with self._lock:
            return self._bindings[name]

    def names(self):
        with self._lock:
            return sorted(self._bindings)

    def search(self, query, limit=10):
        if not isinstance(query, str) or type(limit) is not int or not 0 <= limit <= 100:
            raise ValueError("Search requires a text query and limit between 0 and 100")
        terms = query.casefold().split()
        result = []
        with self._lock:
            for name in sorted(self._bindings):
                spec = self._bindings[name].spec
                if all(term in f"{name} {spec.description}".casefold() for term in terms):
                    result.append({"name": name, "description": spec.description,
                                   "definition_hash": spec.definition_hash,
                                   "network": spec.network, "requires_approval": spec.requires_approval})
        return result[:limit]
