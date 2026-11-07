"""Process-local elision of a legacy stack that TemporalEnv immediately replaces.

No capture/guard/auxiliary code is removed. Keep the legacy deque updated for
reset/resume callers; only avoid its temporary concatenation. Final observations
still own their storage in HybridHistory.observation().
"""
import ast
import inspect
import textwrap


def latest_only(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    matches = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'packed'):
            matches.append(node)
    expected = 'np.concatenate(tuple(self.native_history), axis=0)'
    if len(matches) != 1 or ast.unparse(matches[0].value) != expected:
        raise RuntimeError('Legacy packing implementation changed; revalidate optimization')
    matches[0].value = ast.copy_location(ast.Name(id='frame', ctx=ast.Load()), matches[0].value)
    ast.fix_missing_locations(tree)
    scope = dict(function.__globals__)
    scope.update(inspect.getclosurevars(function).nonlocals)
    exec(compile(tree, '<staged latest-only pack>', 'exec'), scope)
    replacement = scope[function.__name__]
    replacement.__doc__ = function.__doc__
    return replacement


def install(trainer):
    if getattr(trainer, '_staged_latest_only', False):
        return
    original = trainer.build_env_class

    def build():
        cls = original()
        names = [base.__name__ for base in cls.__mro__]
        if 'TemporalEnv' not in names or 'CommandHistoryEnv' not in names:
            raise ValueError('Latest-only intermediate requires temporal + command-history wrappers')
        owners = [base for base in cls.__mro__ if base.__name__ == 'PPOV61Env' and '_pack' in base.__dict__]
        if len(owners) != 1:
            raise ValueError('Unexpected observation inheritance')
        owner = owners[0]
        if not getattr(owner.__dict__['_pack'], '_latest_only', False):
            replacement = latest_only(owner.__dict__['_pack'])
            replacement._latest_only = True
            owner._pack = replacement
        return cls

    trainer.build_env_class = build
    trainer._staged_latest_only = True
