import numpy as np

def buffer(x, n, p=0, opt=None):
    '''Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Args
    ----
    x:   signal array
    n:   number of data segments
    p:   number of values to overlap
    opt: initial condition options. default sets the first `p` values
         to zero, while 'nodelay' begins filling the buffer immediately.
    '''
    if p >= n:
        raise ValueError('p ({}) must be less than n ({}).'.format(p,n))

    # Calculate number of columns of buffer array
    cols = int(np.ceil(len(x)/float(n-p)))
    # Check for opt parameters
    if opt == 'nodelay':
        # Need extra column to handle additional values left
        cols += 1
    elif opt != None:
        raise SystemError('Only `None` (default initial condition) and '
                          '`nodelay` (skip initial condition) have been '
                          'implemented')
    # Create empty buffer array
    b = np.zeros((n, cols))

    # Fill buffer by column handling for initial condition and overlap
    j = 0
    for i in range(cols):
        # Set first column to n values from x, move to next iteration
        if i == 0 and opt == 'nodelay':
            b[0:n,i] = x[0:n]
            continue
        # set first values of row to last p values
        elif i != 0 and p != 0:
            b[:p, i] = b[-p:, i-1]
        # If initial condition, set p elements in buffer array to zero
        else:
            b[:p, i] = 0
        # Get stop index positions for x
        k = j + n - p
        # Get stop index position for b, matching number sliced from x
        n_end = p+len(x[j:k])
        # Assign values to buffer array from x
        b[p:n_end,i] = x[j:k]
        # Update start index location for next iteration of x
        j = k
    return b

def rgb2ycbcr(im):

    im = np.array(im, dtype=int)
    xform = np.array([[.2568, .5041, .0979], [-.1482, -.291, .4392], [.4392, -.3678, -.0714]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    ycbcr[:,:,0] += 16
    return np.uint8(ycbcr)

def generate_skinmap(img):
    
    height, width = img.shape[:-1]
    output = np.zeros([height, width])
    img_ycbcr = rgb2ycbcr(img)
    cb = img_ycbcr[:,:,1]
    cr = img_ycbcr[:,:,2]
    r,c = np.where((cb>=98) & (cb<=142) & (cr>=133) & (cr<=177))
    for i in range(len(r)):
        output[r[i], c[i]] = 1

    return output



