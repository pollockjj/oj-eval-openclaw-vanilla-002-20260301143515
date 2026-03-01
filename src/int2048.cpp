#include "int2048.h"

namespace sjtu {

namespace {

using cd = std::complex<double>;

void fft(std::vector<cd> &a, bool invert) {
  int n = static_cast<int>(a.size());
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      cd tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
  }

  const double PI = std::acos(-1.0);
  for (int len = 2; len <= n; len <<= 1) {
    double ang = 2.0 * PI / static_cast<double>(len);
    if (invert)
      ang = -ang;
    cd wlen(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      cd w(1.0, 0.0);
      int half = len >> 1;
      for (int j = 0; j < half; ++j) {
        cd u = a[i + j];
        cd v = a[i + j + half] * w;
        a[i + j] = u + v;
        a[i + j + half] = u - v;
        w *= wlen;
      }
    }
  }

  if (invert) {
    double invN = 1.0 / static_cast<double>(n);
    for (int i = 0; i < n; ++i)
      a[i] *= invN;
  }
}

std::vector<int> multiply_fft(const std::vector<int> &a, const std::vector<int> &b,
                              int base) {
  if (a.empty() || b.empty())
    return std::vector<int>();

  int need = static_cast<int>(a.size() + b.size());
  int n = 1;
  while (n < need)
    n <<= 1;

  std::vector<cd> fa(n), fb(n);
  for (int i = 0; i < static_cast<int>(a.size()); ++i)
    fa[i] = cd(static_cast<double>(a[i]), 0.0);
  for (int i = 0; i < static_cast<int>(b.size()); ++i)
    fb[i] = cd(static_cast<double>(b[i]), 0.0);

  fft(fa, false);
  fft(fb, false);
  for (int i = 0; i < n; ++i)
    fa[i] *= fb[i];
  fft(fa, true);

  std::vector<int> res(need + 2, 0);
  long long carry = 0;
  for (int i = 0; i < need; ++i) {
    long long cur = static_cast<long long>(fa[i].real() + (fa[i].real() >= 0 ? 0.5 : -0.5)) + carry;
    int digit = static_cast<int>(cur % base);
    if (digit < 0)
      digit += base;
    carry = (cur - digit) / base;
    res[i] = digit;
  }

  int pos = need;
  while (carry > 0) {
    res[pos++] = static_cast<int>(carry % base);
    carry /= base;
  }
  res.resize(pos);

  while (!res.empty() && res.back() == 0)
    res.pop_back();
  return res;
}

} // namespace

bool int2048::isZero() const { return d.size() == 1 && d[0] == 0; }

void int2048::trim() {
  while (d.size() > 1 && d.back() == 0)
    d.pop_back();
  if (isZero())
    neg = false;
}

int int2048::absCmp(const int2048 &a, const int2048 &b) {
  if (a.d.size() != b.d.size())
    return a.d.size() < b.d.size() ? -1 : 1;
  for (int i = static_cast<int>(a.d.size()) - 1; i >= 0; --i) {
    if (a.d[i] != b.d[i])
      return a.d[i] < b.d[i] ? -1 : 1;
  }
  return 0;
}

int2048 int2048::absAdd(const int2048 &a, const int2048 &b) {
  int2048 res;
  int n = static_cast<int>(a.d.size());
  int m = static_cast<int>(b.d.size());
  int len = n > m ? n : m;
  res.d.assign(len + 1, 0);

  int carry = 0;
  for (int i = 0; i < len; ++i) {
    int cur = carry;
    if (i < n)
      cur += a.d[i];
    if (i < m)
      cur += b.d[i];
    if (cur >= BASE) {
      carry = 1;
      cur -= BASE;
    } else {
      carry = 0;
    }
    res.d[i] = cur;
  }
  res.d[len] = carry;
  res.neg = false;
  res.trim();
  return res;
}

int2048 int2048::absSub(const int2048 &a, const int2048 &b) {
  int2048 res;
  int n = static_cast<int>(a.d.size());
  int m = static_cast<int>(b.d.size());
  res.d.assign(n, 0);

  int borrow = 0;
  for (int i = 0; i < n; ++i) {
    int cur = a.d[i] - borrow - (i < m ? b.d[i] : 0);
    if (cur < 0) {
      cur += BASE;
      borrow = 1;
    } else {
      borrow = 0;
    }
    res.d[i] = cur;
  }

  res.neg = false;
  res.trim();
  return res;
}

int2048 int2048::mulInt(int x) const {
  int2048 res;
  if (x == 0 || isZero()) {
    res = int2048(0);
    return res;
  }
  if (x < 0)
    x = -x;

  res.d.assign(d.size() + 5, 0);
  long long carry = 0;
  int pos = 0;
  for (int i = 0; i < static_cast<int>(d.size()); ++i) {
    long long cur = static_cast<long long>(d[i]) * x + carry;
    res.d[pos++] = static_cast<int>(cur % BASE);
    carry = cur / BASE;
  }
  while (carry > 0) {
    res.d[pos++] = static_cast<int>(carry % BASE);
    carry /= BASE;
  }
  res.d.resize(pos);
  res.neg = false;
  res.trim();
  return res;
}

void int2048::mulBaseAdd(int x) {
  if (isZero()) {
    d[0] = x;
    trim();
    return;
  }
  d.push_back(0);
  for (int i = static_cast<int>(d.size()) - 1; i > 0; --i)
    d[i] = d[i - 1];
  d[0] = x;
  trim();
}

int int2048::estimateQuotientDigit(const int2048 &r, const int2048 &b) {
  int rn = static_cast<int>(r.d.size());
  int bn = static_cast<int>(b.d.size());
  if (rn < bn)
    return 0;

  long long highR = r.d[rn - 1];
  long long highB = b.d[bn - 1];
  long long est;
  if (rn == bn) {
    est = highR / highB;
  } else {
    long long nextR = r.d[rn - 2];
    est = (highR * BASE + nextR) / highB;
  }
  if (est >= BASE)
    est = BASE - 1;
  if (est < 0)
    est = 0;
  return static_cast<int>(est);
}

void int2048::divmodAbs(const int2048 &a, const int2048 &b, int2048 &q, int2048 &r) {
  if (absCmp(a, b) < 0) {
    q = int2048(0);
    r = a;
    r.neg = false;
    return;
  }

  const long long B = BASE;
  int n = static_cast<int>(a.d.size());
  int m = static_cast<int>(b.d.size());

  if (m == 1) {
    q.d.assign(n, 0);
    long long rem = 0;
    long long div = b.d[0];
    for (int i = n - 1; i >= 0; --i) {
      long long cur = a.d[i] + rem * B;
      q.d[i] = static_cast<int>(cur / div);
      rem = cur % div;
    }
    q.neg = false;
    q.trim();

    r = int2048(0);
    r.d[0] = static_cast<int>(rem);
    r.neg = false;
    r.trim();
    return;
  }

  int norm = static_cast<int>(B / (b.d.back() + 1));
  std::vector<long long> u(n + 1, 0), v(m, 0);

  long long carry = 0;
  for (int i = 0; i < n; ++i) {
    long long cur = static_cast<long long>(a.d[i]) * norm + carry;
    u[i] = cur % B;
    carry = cur / B;
  }
  u[n] = carry;

  carry = 0;
  for (int i = 0; i < m; ++i) {
    long long cur = static_cast<long long>(b.d[i]) * norm + carry;
    v[i] = cur % B;
    carry = cur / B;
  }

  q.d.assign(n - m + 1, 0);
  q.neg = false;

  for (int j = n - m; j >= 0; --j) {
    long long u2 = u[j + m];
    long long u1 = u[j + m - 1];
    long long u0 = (m >= 2 ? u[j + m - 2] : 0);

    long long v1 = v[m - 1];
    long long v0 = (m >= 2 ? v[m - 2] : 0);

    long long qhat = (u2 * B + u1) / v1;
    long long rhat = (u2 * B + u1) % v1;

    if (qhat >= B) {
      qhat = B - 1;
      rhat += v1;
    }

    while (m >= 2 && qhat * v0 > B * rhat + u0) {
      --qhat;
      rhat += v1;
      if (rhat >= B)
        break;
    }

    long long borrow = 0;
    long long mulCarry = 0;
    for (int i = 0; i < m; ++i) {
      long long prod = qhat * v[i] + mulCarry;
      mulCarry = prod / B;
      prod %= B;

      long long sub = u[j + i] - prod - borrow;
      if (sub < 0) {
        sub += B;
        borrow = 1;
      } else {
        borrow = 0;
      }
      u[j + i] = sub;
    }

    long long subTop = u[j + m] - mulCarry - borrow;
    if (subTop < 0) {
      --qhat;
      long long addCarry = 0;
      for (int i = 0; i < m; ++i) {
        long long sum = u[j + i] + v[i] + addCarry;
        if (sum >= B) {
          sum -= B;
          addCarry = 1;
        } else {
          addCarry = 0;
        }
        u[j + i] = sum;
      }
      u[j + m] += addCarry;
    } else {
      u[j + m] = subTop;
    }

    q.d[j] = static_cast<int>(qhat);
  }

  q.trim();

  r.d.assign(m, 0);
  long long rem = 0;
  for (int i = m - 1; i >= 0; --i) {
    long long cur = u[i] + rem * B;
    r.d[i] = static_cast<int>(cur / norm);
    rem = cur % norm;
  }
  r.neg = false;
  r.trim();
}

void int2048::divmodFloor(const int2048 &a, const int2048 &b, int2048 &q, int2048 &r) {
  int2048 aa = a;
  aa.neg = false;
  int2048 bb = b;
  bb.neg = false;

  divmodAbs(aa, bb, q, r);

  bool qNeg = (a.neg != b.neg);
  if (qNeg && !q.isZero())
    q.neg = true;
  else
    q.neg = false;

  if (a.neg && !r.isZero())
    r.neg = true;
  else
    r.neg = false;

  // Convert trunc division to floor division when signs differ and remainder != 0
  if (!r.isZero() && (a.neg != b.neg)) {
    q -= int2048(1);
    r += b;
  }
  q.trim();
  r.trim();
}

int2048::int2048() : d(1, 0), neg(false) {}

int2048::int2048(long long x) : d(1, 0), neg(false) {
  if (x < 0) {
    neg = true;
    x = -x;
  }
  d.clear();
  if (x == 0)
    d.push_back(0);
  while (x > 0) {
    d.push_back(static_cast<int>(x % BASE));
    x /= BASE;
  }
  trim();
}

int2048::int2048(const std::string &s) : d(1, 0), neg(false) { read(s); }

int2048::int2048(const int2048 &o) : d(o.d), neg(o.neg) {}

void int2048::read(const std::string &s) {
  d.clear();
  neg = false;

  int n = static_cast<int>(s.size());
  int pos = 0;
  if (pos < n && (s[pos] == '-' || s[pos] == '+')) {
    neg = (s[pos] == '-');
    ++pos;
  }

  while (pos < n && s[pos] == '0')
    ++pos;

  if (pos == n) {
    d.push_back(0);
    neg = false;
    return;
  }

  for (int i = n; i > pos; i -= BASE_DIGITS) {
    int l = i - BASE_DIGITS;
    if (l < pos)
      l = pos;
    int x = 0;
    for (int j = l; j < i; ++j)
      x = x * 10 + (s[j] - '0');
    d.push_back(x);
  }

  trim();
}

void int2048::print() { std::cout << *this; }

int2048 &int2048::add(const int2048 &o) {
  *this += o;
  return *this;
}

int2048 add(int2048 a, const int2048 &b) {
  a += b;
  return a;
}

int2048 &int2048::minus(const int2048 &o) {
  *this -= o;
  return *this;
}

int2048 minus(int2048 a, const int2048 &b) {
  a -= b;
  return a;
}

int2048 int2048::operator+() const { return *this; }

int2048 int2048::operator-() const {
  int2048 res(*this);
  if (!res.isZero())
    res.neg = !res.neg;
  return res;
}

int2048 &int2048::operator=(const int2048 &o) {
  if (this == &o)
    return *this;
  d = o.d;
  neg = o.neg;
  return *this;
}

int2048 &int2048::operator+=(const int2048 &o) {
  if (neg == o.neg) {
    int2048 res = absAdd(*this, o);
    res.neg = neg;
    *this = res;
  } else {
    int cmp = absCmp(*this, o);
    if (cmp == 0) {
      *this = int2048(0);
    } else if (cmp > 0) {
      int2048 res = absSub(*this, o);
      res.neg = neg;
      *this = res;
    } else {
      int2048 res = absSub(o, *this);
      res.neg = o.neg;
      *this = res;
    }
  }
  trim();
  return *this;
}

int2048 operator+(int2048 a, const int2048 &b) {
  a += b;
  return a;
}

int2048 &int2048::operator-=(const int2048 &o) {
  *this += (-o);
  return *this;
}

int2048 operator-(int2048 a, const int2048 &b) {
  a -= b;
  return a;
}

int2048 &int2048::operator*=(const int2048 &o) {
  if (isZero() || o.isZero()) {
    *this = int2048(0);
    return *this;
  }

  int n = static_cast<int>(d.size());
  int m = static_cast<int>(o.d.size());
  std::vector<int> prod;

  if ((n > m ? m : n) <= 32) {
    prod.assign(n + m + 2, 0);
    for (int i = 0; i < n; ++i) {
      long long carry = 0;
      for (int j = 0; j < m || carry > 0; ++j) {
        long long cur = prod[i + j] + carry;
        if (j < m)
          cur += 1LL * d[i] * o.d[j];
        prod[i + j] = static_cast<int>(cur % BASE);
        carry = cur / BASE;
      }
    }
    while (!prod.empty() && prod.back() == 0)
      prod.pop_back();
  } else {
    prod = multiply_fft(d, o.d, BASE);
  }

  d = prod;
  neg = (neg != o.neg);
  trim();
  return *this;
}

int2048 operator*(int2048 a, const int2048 &b) {
  a *= b;
  return a;
}

int2048 &int2048::operator/=(const int2048 &o) {
  int2048 q, r;
  divmodFloor(*this, o, q, r);
  *this = q;
  return *this;
}

int2048 operator/(int2048 a, const int2048 &b) {
  a /= b;
  return a;
}

int2048 &int2048::operator%=(const int2048 &o) {
  int2048 q, r;
  divmodFloor(*this, o, q, r);
  *this = r;
  return *this;
}

int2048 operator%(int2048 a, const int2048 &b) {
  a %= b;
  return a;
}

std::istream &operator>>(std::istream &is, int2048 &x) {
  std::string s;
  is >> s;
  x.read(s);
  return is;
}

std::ostream &operator<<(std::ostream &os, const int2048 &x) {
  if (x.neg && !x.isZero())
    os << '-';
  int n = static_cast<int>(x.d.size());
  os << x.d[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    int v = x.d[i];
    char buf[3];
    buf[0] = static_cast<char>('0' + (v / 100));
    buf[1] = static_cast<char>('0' + (v / 10) % 10);
    buf[2] = static_cast<char>('0' + (v % 10));
    os.write(buf, 3);
  }
  return os;
}

bool operator==(const int2048 &a, const int2048 &b) {
  return a.neg == b.neg && a.d == b.d;
}

bool operator!=(const int2048 &a, const int2048 &b) { return !(a == b); }

bool operator<(const int2048 &a, const int2048 &b) {
  if (a.neg != b.neg)
    return a.neg;
  int cmp = int2048::absCmp(a, b);
  if (!a.neg)
    return cmp < 0;
  return cmp > 0;
}

bool operator>(const int2048 &a, const int2048 &b) { return b < a; }

bool operator<=(const int2048 &a, const int2048 &b) { return !(b < a); }

bool operator>=(const int2048 &a, const int2048 &b) { return !(a < b); }

} // namespace sjtu