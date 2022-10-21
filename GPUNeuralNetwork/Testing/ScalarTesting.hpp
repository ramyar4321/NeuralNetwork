#ifndef SCALAR_TESTING
#define SCALAR_TESTING

namespace gpu{
    /**
     * 
     * This class will test important methods of 
     * Scalar class. Tests will focus on success of operations
     * on d_vec. The testing will not be rigorous.
     * 
    */
    class ScalarTesting{

        public:

            ScalarTesting();

            void testCopyConstructor();
            void testEqualOperator();
            void testIsEqualOperator();

    };
}

#endif // End of VECTOR_TESTING