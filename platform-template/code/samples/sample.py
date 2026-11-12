import argparse
from pathlib import Path

import numpy as np
import reference


ROOT = Path(__file__).resolve().parent

OUTPUTS = [
    ('y.bin', [1], np.dtype('float16'), 2, 0.00195, 0.00195, 0.01),
]


def generate(directory):
    rng = np.random.default_rng(42)
    for name in ('input', 'output', 'golden'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    for item in OUTPUTS:
        (directory / 'output' / item[0]).unlink(missing_ok=True)
        (directory / 'golden' / item[0]).unlink(missing_ok=True)
    x0 = np.asarray([-1.1924], dtype=np.dtype('float16')).reshape([1])
    (directory / 'input' / 'x.bin').write_bytes(x0.tobytes())
    result = reference.impl(x0, **{})
    result = (result,)
    values = [result[0]]
    for item, array in zip(OUTPUTS, values):
        name, shape, dtype = item[:3]
        if np.isscalar(array):
            array = np.asarray(array).reshape(shape)
        array = np.asarray(array)
        if array.dtype != dtype or list(array.shape) != shape:
            raise ValueError('reference dtype/shape mismatch: ' + name)
        (directory / 'golden' / name).write_bytes(array.tobytes())
    print('Sample data generated.')


def verify(directory):
    passed = True
    expected = {item[0] for item in OUTPUTS}
    for folder in ('output', 'golden'):
        actual = {p.name for p in (directory / folder).iterdir()}
        if actual != expected:
            print('VERIFY ERROR: unexpected or missing files in ' + folder)
            return 1
    for item in OUTPUTS:
        name, shape, dtype, size, rtol, atol, tol = item[:7]
        try:
            raw = (directory / 'output' / name).read_bytes()
            golden = (directory / 'golden' / name).read_bytes()
            if len(raw) != size or len(golden) != size:
                raise ValueError('exact byte size mismatch')
            a, b = [np.frombuffer(v, dtype=dtype) for v in (raw, golden)]
            close = np.isclose(
                a.astype(np.float64),
                b.astype(np.float64),
                rtol=rtol,
                atol=atol,
                equal_nan=True
            )
            bad = int(np.count_nonzero(~close))
            ok = bad <= tol * a.size
            print(
                ('PASS' if ok else 'Wrong Answer')
                + ': ' + name + f', mismatched={bad}/{a.size}'
            )
            passed &= ok
        except (OSError, ValueError) as error:
            print('VERIFY ERROR: ' + name + ': ' + str(error))
            passed = False
    print('PASSED' if passed else 'FAILED')
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('generate', 'verify'))
    parser.add_argument('--directory', type=Path, default=ROOT / 'build-npu')
    args = parser.parse_args()
    try:
        if args.action == 'generate':
            generate(args.directory)
        else:
            raise SystemExit(verify(args.directory))
    except Exception as error:
        print('SAMPLE ERROR: ' + str(error))
        raise SystemExit(2)
